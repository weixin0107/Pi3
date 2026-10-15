from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
from pathlib import Path
import re
import torch
import numpy as np
from datasets.base.base_dataset import sample_resolutions
import hydra

from pi3.models.loss import Pi3Loss

class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.test_loss)

        export_cfg = cfg.train.get('pointcloud_export', {})
        self.pointcloud_export_enabled = bool(export_cfg.get('enabled', False))
        self.pointcloud_export_dir = Path(
            export_cfg.get('output_dir', '/mnt/data/wangzhixing/tmp/pi3_local_gt')
        )
        self.pointcloud_export_max_sequences = int(export_cfg.get('max_sequences', 1))
        self.pointcloud_exported_sequences = set()
        if hasattr(self.test_loss, 'capture_pointclouds'):
            self.test_loss.capture_pointclouds = (
                self.pointcloud_export_enabled and self.accelerator.is_main_process
            )

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        pred = self.model(imgs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)
            if self.pointcloud_export_enabled and self.accelerator.is_main_process:
                self._export_pointcloud_batch(batch)

        return EasyDict(
            loss=loss,
            **details
        )

    @staticmethod
    def _batch_metadata(view, key, sample_idx):
        values = view.get(key)
        if isinstance(values, (list, tuple)):
            return values[sample_idx] if sample_idx < len(values) else None
        return values

    @staticmethod
    def _safe_path_component(value):
        value = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
        return value or 'unknown'

    def _export_pointcloud_batch(self, batch):
        export = getattr(self.test_loss, 'last_pointcloud_export', None)
        if export is None or not batch:
            return

        predicted = export['pred_local_points']
        ground_truth = export['gt_local_points']
        valid_masks = export['valid_masks']
        images = export['imgs']
        batch_size, num_views = predicted.shape[:2]
        max_sequences = self.pointcloud_export_max_sequences
        xyzrgb_format = ['%.6f', '%.6f', '%.6f', '%d', '%d', '%d']

        for sample_idx in range(batch_size):
            if max_sequences > 0 and len(self.pointcloud_exported_sequences) >= max_sequences:
                break

            label = self._batch_metadata(batch[0], 'label', sample_idx)
            label = str(label) if label is not None else f'sample_{sample_idx:04d}'
            if label in self.pointcloud_exported_sequences:
                continue

            sequence_dir = self.pointcloud_export_dir / self._safe_path_component(label)
            sequence_dir.mkdir(parents=True, exist_ok=True)

            for view_idx in range(min(num_views, len(batch))):
                instance = self._batch_metadata(batch[view_idx], 'instance', sample_idx)
                frame_name = Path(str(instance or f'frame_{view_idx:04d}')).stem
                stem = f'{view_idx:04d}_{self._safe_path_component(frame_name)}'

                mask = valid_masks[sample_idx, view_idx].bool()
                pred_frame = predicted[sample_idx, view_idx]
                gt_frame = ground_truth[sample_idx, view_idx]
                mask &= torch.isfinite(pred_frame).all(dim=-1)
                mask &= torch.isfinite(gt_frame).all(dim=-1)

                rgb_frame = (images[sample_idx, view_idx].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8)
                rgb_values = rgb_frame[mask].numpy()
                pred_xyzrgb = np.concatenate((pred_frame[mask].numpy(), rgb_values), axis=1)
                gt_xyzrgb = np.concatenate((gt_frame[mask].numpy(), rgb_values), axis=1)

                np.savetxt(
                    sequence_dir / f'{stem}_local_points.txt',
                    pred_xyzrgb,
                    fmt=xyzrgb_format,
                )
                np.savetxt(
                    sequence_dir / f'{stem}_gt_points.txt',
                    gt_xyzrgb,
                    fmt=xyzrgb_format,
                )

            self.pointcloud_exported_sequences.add(label)
            self.log_info(f'Exported point clouds for {label} to {sequence_dir}')


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
