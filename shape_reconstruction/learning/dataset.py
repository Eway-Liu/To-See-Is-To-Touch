"""Validated paired data with shared training/inference normalization."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data.pairs import load_pair
from .augmentation import augment_pair


class ShapeReconstructionDataset(Dataset):
    def __init__(self, mode='train', data_dir='./reconstruction_data',
                 transform=None, augmentation=True, manifest=None,
                 height_scale=10.0, geometric=False, input_mode='positive_diff'):
        self.root = Path(data_dir)
        self.transform = transform
        self.augmentation = augmentation and mode == 'train'
        self.height_scale = height_scale
        self.geometric = geometric
        self.input_mode = input_mode
        if height_scale <= 0 or input_mode not in ('positive_diff', 'signed_diff'):
            raise ValueError('Invalid normalization settings')
        if manifest:
            document = json.loads(Path(manifest).read_text(encoding='utf-8'))
            self.names = document['splits'][mode]
            self.label_source = document.get('label_source', 'unknown')
        else:
            if (self.root / mode / 'diff_images').is_dir():
                self.root = self.root / mode
            self.names = sorted(p.name for p in (self.root / 'diff_images').glob('*.npy'))
            self.label_source = 'unknown'
        if not self.names:
            raise ValueError(f'Empty {mode} dataset: {self.root}')
        if len(set(self.names)) != len(self.names):
            raise ValueError('Duplicate sample names in dataset')
        self.diff_images = [str(self.root / 'diff_images' / n) for n in self.names]
        self.height_maps = [str(self.root / 'height_maps' / n) for n in self.names]
        self.shape = None
        for name in self.names:
            diff, _ = load_pair(self.root, name, height_scale)
            if self.shape is not None and diff.shape != self.shape:
                raise ValueError(f'Mixed image shapes in dataset: {name}')
            self.shape = diff.shape

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        diff, height = load_pair(self.root, self.names[idx], self.height_scale)
        if self.augmentation:
            diff, height = augment_pair(diff, height, geometric=self.geometric)
        lower = -255 if self.input_mode == 'signed_diff' else 0
        diff = np.ascontiguousarray(np.clip(diff, lower, 255) / 255, dtype=np.float32)
        height = np.ascontiguousarray(height / self.height_scale, dtype=np.float32)
        x, y = torch.from_numpy(diff[None]), torch.from_numpy(height[None])
        if self.transform:
            # A paired callable must share random geometric parameters.
            x, y = self.transform(x, y)
        return x, y


class TactileDataset(ShapeReconstructionDataset):
    def __init__(self, data_dir, augment=False, **kwargs):
        super().__init__(mode='train' if augment else 'val', data_dir=data_dir,
                         augmentation=augment, **kwargs)
