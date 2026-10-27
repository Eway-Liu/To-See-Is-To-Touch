"""Shared validation for acquisition, screening, and training. Units: mm."""
from pathlib import Path
import json
import numpy as np


def manifest_root(manifest, override=None):
    document = json.loads(Path(manifest).read_text(encoding='utf-8'))
    root = Path(override) if override else Path(document['data_dir'])
    if not override and not root.is_absolute():
        root = Path(manifest).resolve().parent / root
    return root.resolve(), document


def validate_pair(diff, height, height_scale=10.0):
    if diff.ndim != 2 or height.ndim != 2 or diff.shape != height.shape:
        raise ValueError(f'Expected matching 2D arrays, got {diff.shape}, {height.shape}')
    if min(diff.shape) < 16:
        raise ValueError('Images must be at least 16 pixels on each axis')
    if not np.isfinite(diff).all() or not np.isfinite(height).all():
        raise ValueError('Non-finite input or height')
    if diff.min() < -255 or diff.max() > 255:
        raise ValueError('Difference must be in [-255, 255] gray levels')
    if height.min() < -1e-5 or height.max() > height_scale:
        raise ValueError(f'Height outside [0, {height_scale}] mm; check calibration/height_scale')


def load_pair(root, name, height_scale=10.0):
    root = Path(root)
    if Path(name).name != name or not name.endswith('.npy'):
        raise ValueError(f'Invalid sample name: {name}')
    diff = np.load(root / 'diff_images' / name, allow_pickle=False).astype(np.float32)
    height = np.load(root / 'height_maps' / name, allow_pickle=False).astype(np.float32)
    validate_pair(diff, height, height_scale)
    return diff, np.maximum(height, 0)
