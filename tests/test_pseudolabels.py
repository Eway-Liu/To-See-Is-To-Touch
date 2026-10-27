"""Offline pseudo-label generation must preserve pairing and contact boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from shape_reconstruction.data.generate_pseudolabels import generate, generate_synthetic
from shape_reconstruction.data.prepare_dataset import prepare
from shape_reconstruction.learning.evaluation import traditional_from_saved


class PseudolabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        (self.source / 'raw_images').mkdir(parents=True)
        (self.source / 'reference_images').mkdir()
        sensor = self.root / 'calibration' / 'sensor_1'
        (sensor / 'depth_calibration').mkdir(parents=True)
        (sensor / 'camera_calibration').mkdir()
        np.save(sensor / 'depth_calibration' / 'lookup.npy', np.arange(256) / 10)
        np.save(sensor / 'camera_calibration' / 'scale.npy', [0, 0, .05])
        self.config = self.root / 'config.yaml'
        self.config.write_text(yaml.safe_dump({
            'sensor_id': 1, 'calibration_root_dir': 'calibration',
            'camera_calibration': {'camera_calibration_dir': '/camera_calibration',
                                   'position_scale_path': '/scale.npy', 'crop_size': [48, 64]},
            'depth_calibration': {'depth_calibration_dir': '/depth_calibration',
                                  'Pixel_to_Depth_path': '/lookup.npy'},
            'sensor_reconstruction': {'lighting_threshold': 2, 'kernel_list': [7, 7],
                                      'depth_k': 100},
        }), encoding='utf-8')
        self.output = self.root / 'labels'

    def save_input(self, name='frame.npy', raw=None, ref=None):
        raw = np.full((48, 64), 100, np.uint8) if raw is None else raw
        ref = np.full((48, 64), 110, np.uint8) if ref is None else ref
        np.save(self.source / 'raw_images' / name, raw)
        np.save(self.source / 'reference_images' / name, ref)

    def test_generates_labels_without_camera_and_refuses_overwrite(self):
        self.save_input()
        # Opening hardware is a regression: this path must work without a camera.
        with patch('cv2.VideoCapture', side_effect=AssertionError('Opened camera')):
            count = generate(self.source, self.output, self.config,
                             group='touch1', object_id='ball')
            self.assertEqual(count, 1)
            height = np.load(self.output / 'height_maps/frame.npy')
            np.testing.assert_allclose(height, .8, atol=1e-6)
            self.assertEqual(height.dtype, np.float32)
            np.testing.assert_array_equal(np.load(self.output / 'diff_images/frame.npy'), 10)
            np.testing.assert_allclose(traditional_from_saved(self.output, 'frame.npy', self.config), height)
            original = (self.output / 'samples.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):
                generate(self.source, self.output, self.config, group='touch1')
            self.assertEqual((self.output / 'samples.jsonl').read_bytes(), original)
            np.testing.assert_allclose(np.load(self.output / 'height_maps/frame.npy'), height)
        row = json.loads(original)
        self.assertEqual((row['group'], row['object_id']), ('touch1', 'ball'))
        self.assertEqual(row['label_source'], 'traditional_pseudolabel')
        self.assertEqual(row['mm_per_pixel'], .05)

    def test_preserves_metadata_groups_for_screening_and_training(self):
        rows = []
        for i in range(6):
            name = f'{i}.npy'
            raw = np.full((48, 64), 110 + i, np.uint8)
            raw[12:36, 16:48] = 90
            raw[0, 0] = 120 + i
            self.save_input(name, raw, np.full_like(raw, 110 + i))
            rows.append({'file': name, 'group': f'touch{i // 2}', 'object_id': 'ball',
                         'label_source': 'old_label'})
        (self.source / 'samples.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        self.assertEqual(generate(self.source, self.output, self.config), 6)
        self.assertEqual(np.load(self.output / 'diff_images/0.npy')[0, 0], -10)
        self.assertAlmostEqual(float(np.load(self.output / 'height_maps/5.npy')[24, 32]), 2.3, places=5)
        doc = prepare(self.output, self.root / 'splits.json')
        self.assertEqual(doc['screening']['accepted'], 6)
        self.assertEqual(doc['label_source'], ['traditional_pseudolabel'])
        groups = {r['file']: r['group'] for r in doc['samples']}
        split_groups = [{groups[name] for name in names} for names in doc['splits'].values()]
        self.assertEqual([len(group) for group in split_groups], [1, 1, 1])
        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        from shape_reconstruction.learning.dataset import ShapeReconstructionDataset
        dataset = ShapeReconstructionDataset(data_dir=self.output, manifest=self.root / 'splits.json',
                                             augmentation=False)
        self.assertEqual(tuple(dataset[0][0].shape), (1, 48, 64))

    def test_png_inputs_support_a_shared_reference(self):
        raw = np.full((48, 64, 3), 100, np.uint8)
        cv2.imwrite(str(self.source / 'raw_images/contact.png'), raw)
        reference = self.root / 'reference.png'
        cv2.imwrite(str(reference), np.full_like(raw, 110))
        generate(self.source, self.output, self.config, reference=reference, group='touch')
        np.testing.assert_allclose(np.load(self.output / 'height_maps/contact.npy'), .8, atol=1e-6)
        self.assertEqual(np.load(self.output / 'raw_images/contact.npy').shape, (48, 64))

    def test_rejects_missing_group_instead_of_inventing_independent_contacts(self):
        self.save_input()
        with self.assertRaisesRegex(ValueError, 'group'):
            generate(self.source, self.output, self.config)
        self.assertFalse((self.output / 'samples.jsonl').exists())

    def test_rejects_missing_or_mismatched_reference(self):
        self.save_input(ref=np.zeros((32, 32), np.uint8))
        with self.assertRaisesRegex(ValueError, 'shape|matching'):
            generate(self.source, self.output, self.config, group='touch')
        (self.source / 'reference_images/frame.npy').unlink()
        with self.assertRaises(FileNotFoundError):
            generate(self.source, self.output, self.config, group='touch')
        self.assertFalse((self.output / 'samples.jsonl').exists())

    def test_rejects_non_uint8_input_and_duplicate_stems(self):
        self.save_input(raw=np.full((48, 64), 100., np.float32))
        with self.assertRaisesRegex(ValueError, 'uint8'):
            generate(self.source, self.output, self.config, group='touch')
        self.save_input()
        cv2.imwrite(str(self.source / 'raw_images/frame.png'), np.zeros((48, 64), np.uint8))
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
            generate(self.source, self.output, self.config, group='touch')
        self.assertFalse((self.output / 'samples.jsonl').exists())

    def test_empty_input_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, '[Nn]o .*image|[Nn]o .*frame'):
            generate(self.source, self.output, self.config, group='touch')

    def test_synthetic_generation_is_reproducible_and_uses_traditional_labels(self):
        with patch('cv2.VideoCapture', side_effect=AssertionError('Opened camera')):
            self.assertEqual(generate_synthetic(self.output, self.config, num_samples=12,
                                                samples_per_group=2, seed=7), 12)
        second = self.root / 'second'
        third = self.root / 'third'
        generate_synthetic(second, self.config, num_samples=12, samples_per_group=2, seed=7)
        generate_synthetic(third, self.config, num_samples=12, samples_per_group=2, seed=8)
        rows = [json.loads(line) for line in (self.output / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual(len({row['group'] for row in rows}), 6)
        for row in rows:
            name = row['file']
            raw = np.load(self.output / 'raw_images' / name)
            ref = np.load(self.output / 'reference_images' / name)
            diff = np.load(self.output / 'diff_images' / name)
            height = np.load(self.output / 'height_maps' / name)
            self.assertEqual(raw.shape, (48, 64))
            self.assertEqual(raw.dtype, np.uint8)
            self.assertEqual(height.dtype, np.float32)
            self.assertEqual(row['data_source'], 'synthetic')
            self.assertEqual(row['label_source'], 'synthetic_traditional_pseudolabel')
            np.testing.assert_array_equal(diff, ref.astype(np.float32) - raw.astype(np.float32))
            np.testing.assert_array_equal(height, traditional_from_saved(self.output, name, self.config))
            np.testing.assert_array_equal(diff, np.load(second / 'diff_images' / name))
        first = rows[0]['file']
        third_first = sorted((third / 'diff_images').glob('*.npy'))[0]
        self.assertFalse(np.array_equal(np.load(self.output / 'diff_images' / first), np.load(third_first)))
        original = (self.output / 'samples.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):
            generate_synthetic(self.output, self.config, num_samples=12)
        self.assertEqual((self.output / 'samples.jsonl').read_bytes(), original)

    def test_synthetic_groups_survive_screening_and_training(self):
        generate_synthetic(self.output, self.config, num_samples=24, samples_per_group=2, seed=42)
        document = prepare(self.output, self.root / 'synthetic_splits.json')
        self.assertEqual(document['screening']['accepted'], 24)
        self.assertEqual(document['label_source'], ['synthetic_traditional_pseudolabel'])
        groups = {row['file']: row['group'] for row in document['samples']}
        split_groups = [{groups[name] for name in names} for names in document['splits'].values()]
        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        self.assertFalse(split_groups[1] & split_groups[2])
        from shape_reconstruction.learning.dataset import ShapeReconstructionDataset
        dataset = ShapeReconstructionDataset(data_dir=self.output, manifest=self.root / 'synthetic_splits.json',
                                             augmentation=False)
        x, y = dataset[0]
        self.assertEqual(tuple(x.shape), (1, 48, 64))
        self.assertTrue(bool((y >= 0).all()))

    def test_synthetic_rejects_invalid_parameters_before_writing(self):
        for kwargs in ({'num_samples': 0}, {'num_samples': -1}, {'samples_per_group': 0}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate_synthetic(self.output, self.config, **kwargs)
            self.assertFalse(self.output.exists())

    def test_diverse_synthetic_contacts_include_compositions_and_variable_lighting(self):
        generate_synthetic(self.output, self.config, num_samples=80, samples_per_group=5, seed=42)
        rows = [json.loads(line) for line in (self.output / 'samples.jsonl').read_text().splitlines()]
        self.assertTrue(all(row.get('generator_version') == 2 for row in rows))
        components = [c for row in rows[::5] for c in row['components']]
        self.assertEqual({c['family'] for c in components},
                         {'ellipsoid', 'gaussian', 'ridge', 'plateau', 'ring', 'capsule', 'polygon', 'superellipse'})
        self.assertTrue(any(len(row['components']) >= 3 for row in rows))
        self.assertTrue(any(not row['components'] for row in rows))
        centers = np.array([c['center_xy_px'] for c in components])
        self.assertGreater(float(np.ptp(centers[:, 0])), 32)
        self.assertGreater(float(np.ptp(centers[:, 1])), 24)
        self.assertGreater(np.ptp([row['lighting']['base_gray'] for row in rows]), 40)
        for i in range(0, len(rows), 5):
            self.assertTrue(all(row['components'] == rows[i]['components'] for row in rows[i:i+5]))
        # Different procedural shapes must also produce usable, nonconstant labels.
        contacts = [np.load(self.output / 'height_maps' / row['file']) for row in rows if row['components']]
        self.assertTrue(all(height.std() > .001 for height in contacts))

    def test_append_preserves_existing_data_and_extends_original_splits(self):
        generate_synthetic(self.output, self.config, num_samples=12, samples_per_group=2)
        manifest = self.root / 'splits.json'
        original = prepare(self.output, manifest)
        original_files = {p: p.read_bytes() for p in self.output.rglob('*.npy')}
        original_metadata = (self.output / 'samples.jsonl').read_bytes()
        self.assertEqual(generate_synthetic(self.output, self.config, num_samples=12,
                                           samples_per_group=2, append=True), 12)
        self.assertTrue((self.output / 'samples.jsonl').read_bytes().startswith(original_metadata))
        for path, content in original_files.items():
            self.assertEqual(path.read_bytes(), content)
        rows = [json.loads(line) for line in (self.output / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row['group'] for row in rows}), 12)
        self.assertFalse({r['group'] for r in rows[:12]} & {r['group'] for r in rows[12:]})
        extended = prepare(self.output, manifest, extend=True)
        self.assertEqual(extended['screening']['accepted'], 24)
        for split in ('train', 'val', 'test'):
            self.assertTrue(set(original['splits'][split]) <= set(extended['splits'][split]))
        current = {row['file']: row['group'] for row in extended['samples']}
        split_groups = [{current[n] for n in extended['splits'][s]} for s in ('train', 'val', 'test')]
        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        self.assertFalse(split_groups[1] & split_groups[2])

    def test_append_rejects_an_incomplete_dataset_without_writing(self):
        generate_synthetic(self.output, self.config, num_samples=6)
        original = (self.output / 'samples.jsonl').read_bytes()
        next((self.output / 'height_maps').glob('*.npy')).unlink()
        with self.assertRaisesRegex(ValueError, '[Ii]ncomplete|[Mm]issing'):
            generate_synthetic(self.output, self.config, num_samples=5, append=True)
        self.assertEqual((self.output / 'samples.jsonl').read_bytes(), original)

    def test_extending_splits_refuses_changed_original_samples(self):
        generate_synthetic(self.output, self.config, num_samples=12, samples_per_group=2)
        manifest = self.root / 'splits.json'
        original = prepare(self.output, manifest)
        original_manifest = manifest.read_bytes()
        path = self.output / 'diff_images' / original['splits']['train'][0]
        diff = np.load(path)
        diff[0, 0] += 1
        np.save(path, diff)
        with self.assertRaisesRegex(ValueError, '[Cc]hanged|[Mm]odified'):
            prepare(self.output, manifest, extend=True)
        self.assertEqual(manifest.read_bytes(), original_manifest)


if __name__ == '__main__':
    unittest.main()
