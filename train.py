"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from typing import Dict

from tqdm import tqdm
import sys
import math
import torch
import torch.amp

from src.data import CocoEvaluator
from config import Config


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))



class Trainer(object):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = cfg.model
        self.model.to(self.device)
        self.criterion = self.to(cfg.criterion, self.device)
        self.postprocessor = self.to(cfg.postprocessor, self.device)
        self.ema = self.to(cfg.ema, self.device)
        self.scaler = cfg.scaler
        self.last_epoch = cfg.last_epoch
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = cfg.optimizer
        self.lr_scheduler = cfg.lr_scheduler
        self.lr_warmup_scheduler = cfg.lr_warmup_scheduler
        self.train_dataloader = cfg.train_dataloader
        self.val_dataloader = cfg.val_dataloader
        self.evaluator = cfg.evaluator

        self.setup()

    def setup(self):
        self.model.train()
        self.criterion.train()

        for param in self.model.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in self.criterion.parameters():
             param.data = param.data.float()
             param.requires_grad = True

    def to(self, module, device):
        return module.to(device) if hasattr(module, 'to') else module

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = self.model(v)
                state[k] = v.state_dict()

        return state
    
    def train_one_epoch(self,
                    epoch: int, 
                    max_norm: float = 0):
                    
        loss_value = 0.0

        for i, (images, targets) in enumerate(tqdm(self.train_dataloader, desc=f"Training {epoch} Epoch: {loss_value}", leave=True)):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            print(targets)
            global_step = epoch * len(self.train_dataloader) + i
            metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(self.train_dataloader))

            with torch.autocast(device_type=str(self.device), cache_enabled=True):
                outputs = self.model(images, targets=targets)

            with torch.autocast(device_type=str(self.device), enabled=False):
                loss_dict = self.criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            self.scaler.scale(loss).backward()

            if max_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            self.ema.update(self.model)
            self.lr_warmup_scheduler.step()

            loss_value = sum(loss_dict.values())
            print(f"Loss: {loss_value:.4f}")

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

    def fit(self, epoches):
        args = self.cfg
        top1 = 0
        best_stat = {'epoch': -1, }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, epoches):

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = self.train_one_epoch(epoch, max_norm=args.clip_max_norm)

            print(train_stats)

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            if coco_evaluator is not None:
                (self.output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name)

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
        print(test_stats)
        return
    

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    iou_types = coco_evaluator.iou_types
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator


if __name__ == '__main__':
    cfg = Config()
    solver = Trainer(cfg)
    # solver.val()
    solver.fit(10)
