"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from src.nn.backbone.hgnetv2 import HGNetv2
from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.hybrid_encoder import HybridEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

from src.core.workspace import create
from src.core.yaml_utils import load_config, merge_config, merge_dict


class Config:
    def __init__(self, cfg_path: str, **kwargs) -> None:
        # instance / function
        self._model :nn.Module = None
        self._postprocessor :nn.Module = None
        self._criterion :nn.Module = None
        self._optimizer :Optimizer = None
        self._lr_scheduler :LRScheduler = None
        self._lr_warmup_scheduler: LRScheduler = None
        self._train_dataloader :DataLoader = None
        self._val_dataloader :DataLoader = None
        self._ema :nn.Module = None
        self._scaler :GradScaler = None
        self._train_dataset :Dataset = None
        self._val_dataset :Dataset = None
        self._collate_fn :Callable = None
        self._evaluator :Callable[[nn.Module, DataLoader, str], ] = None
        self._writer: SummaryWriter = None

        # dataset
        self.num_workers :int = 0
        self.batch_size :int = None
        self._train_batch_size :int = None
        self._val_batch_size :int = None
        self._train_shuffle: bool = None
        self._val_shuffle: bool = None

        # runtime
        self.resume :str = None
        self.tuning :str = None

        self.epoches :int = 50
        self.last_epoch :int = -1

        self.use_amp :bool = False
        self.use_ema :bool = False
        self.ema_decay :float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn :bool = False
        self.clip_max_norm : float = 0.
        self.find_unused_parameters :bool = None

        self.seed :int = 0
        self.checkpoint_freq :int = 1
        self.output_dir :str = "./output"
        self.device : str = ''
        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        # self._model = DFINE(
        #     backbone=HGNetv2(name="B5", return_idx=[1, 2, 3], freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True), 
        #     decoder=DFINETransformer(feat_channels=[384, 384, 384], reg_scale=8), 
        #     encoder=HybridEncoder(hidden_dim=384, dim_feedforward=2048))
        return self._model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return self._postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return self._criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return self._optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return self._lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return self._lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return self._train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return self._val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return self._ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
            print(type(self._scaler))
        return self._scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from .src.data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return self._evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from src.misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        print(f'building {name} with batch_size={bs}...')
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader
