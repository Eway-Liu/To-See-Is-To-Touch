"""Shared validation for pseudo-label generation, screening, and training. Units: mm."""
from pathlib import Path
import json
import os
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


def save_sample(root,index,diff,height,raw,reference,metadata):
    validate_pair(diff,height)
    stem=f'{index:06d}' if isinstance(index,int) else str(index)
    if Path(stem).name!=stem or stem in ('','.','..') or any(c in stem for c in '/\\:'):
        raise ValueError('Sample must be a safe filename stem')
    if raw.shape!=diff.shape or reference.shape!=diff.shape or raw.dtype!=np.uint8 or reference.dtype!=np.uint8:
        raise ValueError('Raw/reference must be matching uint8 grayscale arrays')
    name=f'{stem}.npy';written=[]
    try:
        for folder,array in (('diff_images',diff),('height_maps',height),
                             ('raw_images',raw),('reference_images',reference)):
            path=Path(root)/folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                written.append(path)
                np.save(stream,array,allow_pickle=False)
        row={'file':name,**metadata}
        with (Path(root)/'samples.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
    except Exception:
        for path in written:path.unlink(missing_ok=True)
        raise
    return name
