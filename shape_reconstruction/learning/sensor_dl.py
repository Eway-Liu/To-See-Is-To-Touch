"""Neural sensor using the same checkpoint and normalization as evaluation."""
import numpy as np
from ..core.sensor import Sensor
from .inference import ReconstructionModel


class SensorDL(Sensor):
    def __init__(self, cfg, model_path=None, use_dl=True, device=None, calibrated=True, ref=None):
        super().__init__(cfg, calibrated=calibrated, ref=ref)
        self.dl_model = None
        self.use_dl = False
        self.device = device
        if model_path:
            self.load_dl_model(model_path)
        if use_dl:
            self.set_method(True)

    def load_dl_model(self, model_path):
        # Never silently label a traditional fallback as neural reconstruction.
        self.dl_model = ReconstructionModel(model_path, self.device)
        self.use_dl = True

    def raw_image_2_height_map_dl(self, img_GRAY):
        if self.dl_model is None:
            raise RuntimeError('Neural checkpoint is not loaded')
        diff = self.ref_GRAY.astype(np.float32) - img_GRAY.astype(np.float32)
        return self.dl_model.predict(diff)

    def raw_image_2_height_map(self, img_GRAY):
        if self.use_dl:
            return self.raw_image_2_height_map_dl(img_GRAY)
        return super().raw_image_2_height_map(img_GRAY)

    def set_method(self, use_dl):
        if use_dl and self.dl_model is None:
            raise RuntimeError('Neural checkpoint is not loaded')
        self.use_dl = bool(use_dl)
        return True

    def get_current_method(self):
        return 'deep_learning' if self.use_dl else 'traditional'
