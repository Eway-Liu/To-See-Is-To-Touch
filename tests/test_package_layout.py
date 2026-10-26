"""Regression checks for configuration paths and lightweight package imports."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shape_reconstruction.config import DEFAULT_CONFIG_PATH, PACKAGE_DIR, load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_resolves_existing_calibration_files(self):
        cfg = load_config()
        self.assertEqual(DEFAULT_CONFIG_PATH.parent, PACKAGE_DIR / 'configs')
        self.assertEqual(Path(cfg['calibration_root_dir']), PACKAGE_DIR / 'calibration')
        sensor_dir = Path(cfg['calibration_root_dir']) / f"sensor_{cfg['sensor_id']}"
        for section, filenames in (
            ('camera_calibration', ('row_index_path', 'col_index_path', 'position_scale_path')),
            ('depth_calibration', ('Pixel_to_Depth_path',)),
        ):
            directory = str(sensor_dir) + cfg[section][f'{section}_dir']
            for key in filenames:
                with self.subTest(key=key):
                    self.assertTrue(Path(directory + cfg[section][key]).is_file())

    def test_default_config_is_independent_of_working_directory(self):
        original = Path.cwd()
        expected = load_config()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_config(), expected)
            finally:
                os.chdir(original)

    def test_custom_relative_root_is_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'configs' / 'sensor.yaml'
            config.parent.mkdir()
            config.write_text('calibration_root_dir: ../calibration\n', encoding='utf-8')
            cfg = load_config(config)
            self.assertEqual(Path(cfg['calibration_root_dir']), (Path(directory) / 'calibration').resolve())

    def test_custom_absolute_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'calibration'
            config = Path(directory) / 'sensor.yaml'
            config.write_text(f'calibration_root_dir: {root}\n', encoding='utf-8')
            self.assertEqual(Path(load_config(config)['calibration_root_dir']), root.resolve())


class PackageTests(unittest.TestCase):
    def test_public_exports_resolve_to_relocated_modules(self):
        import shape_reconstruction
        for name, module in (
            ('Camera', '.core.camera'),
            ('Sensor', '.core.sensor'),
            ('Visualizer', '.visualization.visualizer'),
        ):
            with self.subTest(name=name):
                expected = object()
                with patch('shape_reconstruction.import_module',
                           return_value=SimpleNamespace(**{name: expected})) as importer:
                    try:
                        self.assertIs(getattr(shape_reconstruction, name), expected)
                        self.assertIs(getattr(shape_reconstruction, name), expected)
                        importer.assert_called_once_with(module, 'shape_reconstruction')
                    finally:
                        shape_reconstruction.__dict__.pop(name, None)

    def test_package_import_does_not_load_hardware_or_learning_dependencies(self):
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys, shape_reconstruction; '
             'assert shape_reconstruction.__all__ == ["Camera", "Sensor", "Visualizer"]; '
             'assert not {"cv2", "open3d", "torch"}.intersection(sys.modules)'],
            cwd=PACKAGE_DIR.parent, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_public_attribute_raises_attribute_error(self):
        import shape_reconstruction
        with self.assertRaises(AttributeError):
            getattr(shape_reconstruction, 'does_not_exist')


if __name__ == '__main__':
    unittest.main()
