"""Single checkpoint contract for training, evaluation, and live inference."""
import numpy as np
import torch
import torch.nn.functional as F
from .model import build_model


class ReconstructionModel:
    def __init__(self, model_path, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' not in checkpoint:
            from .legacy_model import UNet
            self.model = UNet()
            self.model.load_state_dict(checkpoint)
            self.model_type = 'legacy_unet'
            self.height_scale = 10.0
            self.input_mode = 'positive_diff'
        else:
            self.model_type = checkpoint.get('model_type', 'unet')
            self.model = build_model(self.model_type, checkpoint.get('base_features', 32))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.height_scale = float(checkpoint.get('height_scale', 10.0))
            self.input_mode = checkpoint.get('input_mode', 'positive_diff')
        if not np.isfinite(self.height_scale) or self.height_scale <= 0 or self.input_mode not in ('positive_diff', 'signed_diff'):
            raise ValueError('Invalid checkpoint normalization')
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, diff_image):
        diff = np.asarray(diff_image, dtype=np.float32)
        if diff.ndim != 2 or min(diff.shape) < 16 or not np.isfinite(diff).all():
            raise ValueError('Expected a finite 2D difference image, at least 16x16')
        lower = -255 if self.input_mode == 'signed_diff' else 0
        x = torch.from_numpy(np.ascontiguousarray(np.clip(diff, lower, 255) / 255))[None, None]
        x = x.to(self.device)
        h, w = diff.shape
        # Historical models require divisible dimensions; crop padding after inference.
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode='replicate')
        y = self.model(x)[0, 0, :h, :w].float().cpu().numpy() * self.height_scale
        if not np.isfinite(y).all():
            raise RuntimeError('Model produced non-finite heights')
        return np.clip(y, 0, self.height_scale).astype(np.float32)
