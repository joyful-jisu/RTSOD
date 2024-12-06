"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from collections import Counter
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from ema import ModelEMA
from src.data.dataloader import BatchImageCollateFunction
from src.data.coco_dataset import CocoDetection
from src.data.coco_eval import CocoEvaluator
from src.data.transforms._transforms import ConvertBoxes, ConvertPILImage
from src.data.transforms.container import Compose
from src.nn.backbone.hgnetv2 import HGNetv2
from src.dfine.dfine import DFINE
from src.dfine.dfine_criterion import DFINECriterion
from src.dfine.dfine_decoder import DFINETransformer
from src.dfine.hybrid_encoder import HybridEncoder
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from src.dfine.matcher import HungarianMatcher
from src.dfine.postprocessor import DFINEPostProcessor
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from src.warmup import LinearWarmup


class Config:
    def __init__(self) -> None:
        self.model = DFINE(
            backbone=HGNetv2(name="B5", 
                            return_idx=[1, 2, 3], 
                            freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True), 
                            decoder=DFINETransformer(feat_channels=[384, 384, 384], reg_scale=8), 
                            encoder=HybridEncoder(hidden_dim=384, dim_feedforward=2048))

        self._postprocessor :nn.Module = None
        self._criterion :nn.Module = None
        self._optimizer :optim.Optimizer = None
        self._lr_scheduler :lr_scheduler.LRScheduler = None
        self._lr_warmup_scheduler: lr_scheduler.LRScheduler = None
        self._train_dataloader :DataLoader = None
        self._val_dataloader :DataLoader = None
        self._ema :nn.Module = None
        self._scaler :torch.GradScaler = None
        self._train_dataset :torch.utils.data.Dataset = None
        self._val_dataset :torch.utils.data.Dataset = None
        self._collate_fn :Callable = None
        self._evaluator :Callable[[nn.Module, DataLoader, str], ] = None

        # dataset
        self.num_workers :int = 0
        self.batch_size :int = 1
        self._train_shuffle: bool = None
        self._val_shuffle: bool = None

        self.epoches :int = 50
        self.last_epoch :int = -1

        self.use_amp :bool = True
        self.use_ema :bool = False
        self.ema_decay :float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn :bool = False
        self.clip_max_norm : float = 0.

        self.seed :int = 0
        self.checkpoint_freq :int = 1
        self.output_dir :str = "./output"
        self.device : str = "cuda:0"
        self.data_source=CocoDetection(
            img_folder="./dataset/images/train",
            ann_file="./dataset/annotations/instances_train.json",
            return_masks=False,
            transforms=Compose(
                ops=[
                    T.RandomPhotometricDistort(p=0.5),
                    T.RandomZoomOut(fill=0),
                    T.RandomIoUCrop(),
                    T.SanitizeBoundingBoxes(min_size=1),
                    T.RandomHorizontalFlip(),
                    T.Resize(size=[640, 640], ),
                    T.SanitizeBoundingBoxes(min_size=1),
                    ConvertPILImage(dtype='float32', scale=True),
                    ConvertBoxes(fmt='cxcywh', normalize=True),
            ],
            policy={'name': 'stop_epoch', 
                    'epoch': 72, 
                    'ops': ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']}),
        ) 
        self.val_data_source=CocoDetection(
            img_folder="./dataset/images/val",
            ann_file="./dataset/annotations/instances_val.json",
            return_masks=False,
            transforms=Compose(ops=[
                T.Resize(size=[640, 640]),
                ConvertPILImage(dtype='float32', scale=True),
            ]),
        ) 

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None:
            self._postprocessor = DFINEPostProcessor(
                num_classes=12,
                num_top_queries=True,
                remap_mscoco_category=False,
                use_focal_loss=True,
            )

        return self._postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None:
            self._criterion = DFINECriterion(
                matcher=HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                         alpha=0.25, gamma=2.0),
                losses=['vfl', 'boxes', 'local'],
                weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_fgl': 0.15, 'loss_ddf': 1.5},
                alpha=0.75)
        return self._criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = AdamW(params=self.model.parameters(), lr=0.00025, 
                                    weight_decay=0.000125, amsgrad=False, 
                                    betas=(0.9, 0.999), eps=1e-8)
        return self._optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None:
            self._lr_scheduler = lr_scheduler.MultiStepLR(
                optimizer=self._optimizer, 
                milestones=Counter({500: 1}),
                gamma=0.1,
                last_epoch=self.last_epoch,
                verbose=False,
            )
        return self._lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None:
            self._lr_warmup_scheduler = LinearWarmup(lr_scheduler=self._lr_scheduler, warmup_duration=500)
        return self._lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                dataset=self.data_source,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=False,
                pin_memory_device="",
                timeout=0,
                worker_init_fn=None,
                multiprocessing_context=None,
                batch_sampler=BatchSampler(RandomSampler(self.data_source), batch_size=self.batch_size, drop_last=True),
                generator=None,
                collate_fn=BatchImageCollateFunction(),
                persistent_workers=False,
            )
        return self._train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                dataset=self.val_data_source,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=False,
                pin_memory_device="",
                timeout=0,
                worker_init_fn=None,
                shuffle=False,
                multiprocessing_context=None,
                generator=None,
                collate_fn=BatchImageCollateFunction(),
                persistent_workers=False,
            )
        return self._val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None:
            self._ema = ModelEMA(model=self.model, decay=0.9999, warmups=1000, start=0)
        return self._ema

    @property
    def scaler(self, ):
        if self._scaler is None:
            self._scaler = torch.GradScaler(enabled=True, 
                                            device="cuda:0", 
                                            backoff_factor=0.5, 
                                            growth_interval=2000,
                                            init_scale=65536.0,
                                            growth_factor=2.0)
        return self._scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None:
            from src.data import get_coco_api_from_dataset
            base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
            self._evaluator = CocoEvaluator(coco_gt=base_ds, iou_types=['bbox'])
        return self._evaluator