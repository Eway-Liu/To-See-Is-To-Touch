"""Load sensor configuration and resolve calibration paths beside the YAML file."""
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / 'configs' / 'shape_config.yaml'


def load_config(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).resolve()
    with config_path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    calibration_root = Path(cfg['calibration_root_dir'])
    if not calibration_root.is_absolute():
        calibration_root = config_path.parent / calibration_root
    cfg['calibration_root_dir'] = str(calibration_root.resolve())
    return cfg
