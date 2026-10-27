"""Paired augmentation without changing the physical height scale."""
import numpy as np


def augment_pair(diff, height, geometric=False, noise_std=0.5):
    # Sensor illumination is directional: geometric transforms require validation.
    if geometric:
        if np.random.random() < 0.5:
            diff, height = np.fliplr(diff), np.fliplr(height)
        if np.random.random() < 0.5:
            diff, height = np.flipud(diff), np.flipud(height)
        choices = (0, 1, 2, 3) if diff.shape[0] == diff.shape[1] else (0, 2)
        k = np.random.choice(choices)
        diff, height = np.rot90(diff, k), np.rot90(height, k)
    if noise_std > 0:
        diff = diff + np.random.normal(0, noise_std, diff.shape).astype(np.float32)
    return np.ascontiguousarray(diff, dtype=np.float32), np.ascontiguousarray(height, dtype=np.float32)
