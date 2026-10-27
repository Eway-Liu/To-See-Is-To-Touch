"""Generate traditional pseudo-labels from random synthetic or saved contact images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..config import DEFAULT_CONFIG_PATH, load_config
from ..core.sensor import Sensor
from .pairs import save_sample


IMAGE_SUFFIXES = {'.npy', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _pixel_scale(cfg):
    directory = Path(cfg['calibration_root_dir']) / f"sensor_{cfg['sensor_id']}"
    cc = cfg['camera_calibration']
    scale = float(np.load(str(directory) + cc['camera_calibration_dir'] + cc['position_scale_path'],
                          allow_pickle=False)[2])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid mm_per_pixel calibration')
    return scale


def read_gray(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = (np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy'
             else cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if image is None or image.dtype != np.uint8:
        raise ValueError(f'{path}: expected a uint8 image')
    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if image.ndim != 2 or min(image.shape) < 16:
        raise ValueError(f'{path}: expected a grayscale image at least 16 pixels on each axis')
    return image


def generate(input_dir, output_dir, config=None, reference=None, group=None, object_id=None):
    """Use saved metadata or an explicit contact group; never infer independent touches."""
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    files = sorted(p for p in (source / 'raw_images').iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f'No input images in {source / "raw_images"}')
    names = [p.stem + '.npy' for p in files]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate image stems would overwrite output samples')
    metadata = {}
    if (source / 'samples.jsonl').is_file():
        for line in (source / 'samples.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['file'] in metadata:
                raise ValueError(f'Duplicate metadata for {row["file"]}')
            metadata[row['file']] = row
    recorded = set()
    if (output / 'samples.jsonl').is_file():
        recorded = {json.loads(line)['file'] for line in
                    (output / 'samples.jsonl').read_text(encoding='utf-8').splitlines()}
    rows = []
    for path, name in zip(files, names):
        if name in recorded or any((output / folder / name).exists() for folder in
                                  ('diff_images', 'height_maps', 'raw_images', 'reference_images')):
            raise FileExistsError(f'Output sample already exists: {name}; use a new output directory')
        row = dict(metadata.get(path.name, metadata.get(name, {})))
        row.pop('file', None)
        if group is not None:
            row['group'] = group
        if not isinstance(row.get('group'), str) or not row['group'].strip():
            raise ValueError(f'{path.name}: supply contact group metadata or --group; do not treat adjacent frames as independent contacts')
        if object_id is not None:
            row['object_id'] = object_id
        row.setdefault('object_id', 'unknown')
        rows.append(row)
    cfg = load_config(config) if config else load_config()
    scale = _pixel_scale(cfg)
    shared_ref = read_gray(reference) if reference is not None else None
    sensor = None
    for path, name, row in zip(files, names, rows):
        raw = read_gray(path)
        ref = shared_ref if shared_ref is not None else read_gray(source / 'reference_images' / path.name)
        if raw.shape != ref.shape:
            raise ValueError(f'{path.name}: raw/reference must have matching shapes')
        if sensor is None:
            sensor = Sensor.from_reference(cfg, ref)
        else:
            sensor.ref_GRAY = ref
        height = sensor.raw_image_2_height_map(raw).astype(np.float32)
        diff = ref.astype(np.float32) - raw.astype(np.float32)
        row.update(label_source='traditional_pseudolabel', mm_per_pixel=scale,
                   max_height_mm=float(height.max()), is_background=bool(height.max() <= .05),
                   lighting_threshold=sensor.lighting_threshold, kernel_list=list(sensor.kernel_list))
        save_sample(output, path.stem, diff, height, raw, ref, row)
    print(f'Generated {len(files)} traditional pseudo-labels in {output}')
    return len(files)


def _contact_profile(u, v, component):
    family = component['family']
    radial = u * u + v * v
    exponent = component['exponent']
    if family == 'ellipsoid':
        return np.clip(1 - radial, 0, 1) ** exponent
    if family == 'gaussian':
        return np.exp(-component['sharpness'] * radial) * (radial < 2.25)
    if family == 'ridge':
        return np.clip(1 - np.abs(u), 0, 1) ** exponent * np.clip(1 - v * v, 0, 1)
    if family == 'plateau':
        return np.clip(component['sharpness'] * (1 - np.maximum(np.abs(u), np.abs(v))), 0, 1)
    if family == 'ring':
        return np.clip(1 - ((np.sqrt(radial) - component['ring_radius']) / component['ring_width']) ** 2, 0, 1)
    if family == 'capsule':
        distance = np.maximum(np.abs(u) - component['segment'], 0) ** 2 + v * v
        return np.clip(1 - distance / component['cap_radius'] ** 2, 0, 1) ** exponent
    if family == 'polygon':
        distance = np.maximum.reduce([u * np.cos(a) + v * np.sin(a)
                                      for a in np.arange(component['sides']) * 2 * np.pi / component['sides']])
    else:  # Superellipses include rounded rectangles and diamond-like contacts.
        power = component['power']
        distance = (np.abs(u) ** power + np.abs(v) ** power) ** (1 / power)
    return np.clip(1 - distance, 0, 1) ** exponent


def _append_offsets(output, cfg, scale, shape):
    rows = [json.loads(line) for line in (output / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    names = {row['file'] for row in rows}
    if not rows or len(names) != len(rows) or any(row.get('data_source') != 'synthetic' or not row.get('group') for row in rows):
        raise ValueError('Append requires complete synthetic metadata with unique sample names and contact groups')
    for folder in ('diff_images', 'height_maps', 'raw_images', 'reference_images'):
        if {p.name for p in (output / folder).glob('*.npy')} != names:
            raise ValueError(f'Incomplete dataset: missing or unrecorded samples in {folder}')
    rc = cfg['sensor_reconstruction']
    if any(not np.isclose(row['mm_per_pixel'], scale) or row['lighting_threshold'] != rc['lighting_threshold']
           or row['kernel_list'] != list(rc['kernel_list']) for row in rows):
        raise ValueError('Append requires the existing calibration scale and reconstruction settings')
    if read_gray(output / 'raw_images' / rows[0]['file']).shape != shape:
        raise ValueError('Append requires the existing image dimensions')
    return len(rows), len({row['group'] for row in rows})


def generate_synthetic(output_dir, config=None, num_samples=120, samples_per_group=5, seed=42, append=False):
    """Generate new procedural frames, optionally appending to an existing synthetic dataset."""
    if num_samples <= 0 or samples_per_group <= 0:
        raise ValueError('num_samples and samples_per_group must be positive')
    np.random.SeedSequence(seed)  # Validate before creating any files.
    cfg = load_config(config) if config else load_config()
    height_px, width_px = cfg['camera_calibration']['crop_size']
    if min(height_px, width_px) < 16:
        raise ValueError('Synthetic images must be at least 16 pixels on each axis')
    shape = (height_px, width_px)
    scale = _pixel_scale(cfg)
    sensor = Sensor.from_reference(cfg, np.zeros(shape, np.uint8))
    max_contrast = min(80, sensor.max_index - sensor.lighting_threshold)
    if max_contrast < 5:
        raise ValueError('Depth lookup needs at least five usable contrast levels')
    output = Path(output_dir).resolve()
    if append:
        sample_offset, group_offset = _append_offsets(output, cfg, scale, shape)
    else:
        output.mkdir(parents=True, exist_ok=False)
        sample_offset = group_offset = 0
    yy, xx = np.indices(shape, dtype=np.float32)
    nx, ny = xx / width_px - .5, yy / height_px - .5
    families = ('ellipsoid', 'gaussian', 'ridge', 'plateau', 'ring', 'capsule', 'polygon', 'superellipse')
    for start in range(0, num_samples, samples_per_group):
        group_index = group_offset + start // samples_per_group
        # Per-group streams keep appended batches reproducible without replaying old samples.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, group_index]))
        background = (group_index + 1) % 10 == 0
        count = 0 if background else (1 if group_index % 3 == 0 else int(rng.integers(2, 5)))
        components = []
        for j in range(count):
            center = rng.uniform(.02, .98, 2) * [width_px, height_px]
            if j and rng.random() < .5:
                center = np.clip(np.array(components[0]['center_xy_px']) + rng.normal(0, .12, 2) * [width_px, height_px],
                                 [.01 * width_px, .01 * height_px], [.99 * width_px, .99 * height_px])
            components.append({'family': families[group_index % len(families)] if j == 0 else str(rng.choice(families)),
                               'center_xy_px': center.tolist(),
                               'radii_xy_px': (rng.uniform(.05, .26, 2) * [width_px, height_px]).tolist(),
                               'angle_rad': float(rng.uniform(0, 2 * np.pi)), 'contrast_gray': float(rng.uniform(.25, .85) * max_contrast),
                               'exponent': float(rng.uniform(.35, 1.8)), 'sharpness': float(rng.uniform(2, 7)),
                               'ring_radius': float(rng.uniform(.5, .8)), 'ring_width': float(rng.uniform(.15, .3)),
                               'segment': float(rng.uniform(.35, .7)), 'cap_radius': float(rng.uniform(.35, .65)),
                               'sides': int(rng.integers(3, 8)), 'power': float(rng.uniform(.8, 5))})
        blend = str(rng.choice(['maximum', 'additive']))
        lighting = {'base_gray': float(rng.uniform(110, 215)), 'gradient_xy': rng.uniform(-35, 35, 2).tolist(),
                    'vignette_gray': float(rng.uniform(0, 25)), 'texture_gray': float(rng.uniform(0, 1.5)),
                    'noise_std': float(rng.uniform(.25, .65))}
        illumination = (lighting['base_gray'] + lighting['gradient_xy'][0] * nx + lighting['gradient_xy'][1] * ny
                        - lighting['vignette_gray'] * (nx * nx + ny * ny)
                        + lighting['texture_gray'] * np.sin(rng.uniform(5, 25) * nx + rng.uniform(5, 25) * ny))
        reference = np.clip(np.rint(illumination + rng.normal(0, .5, shape)), 0, 255).astype(np.uint8)
        sensor.ref_GRAY = reference
        for local_index in range(start, min(start + samples_per_group, num_samples)):
            shift = rng.normal(0, .7, 2)
            size_gain, contrast_gain = rng.uniform(.97, 1.03), rng.uniform(.9, 1.)
            signal = np.zeros(shape, np.float32)
            for component in components:
                dx = xx - component['center_xy_px'][0] - shift[0]
                dy = yy - component['center_xy_px'][1] - shift[1]
                angle = component['angle_rad']
                u = (np.cos(angle) * dx + np.sin(angle) * dy) / (component['radii_xy_px'][0] * size_gain)
                v = (-np.sin(angle) * dx + np.cos(angle) * dy) / (component['radii_xy_px'][1] * size_gain)
                patch = _contact_profile(u, v, component) * component['contrast_gray'] * contrast_gain
                signal = np.maximum(signal, patch) if blend == 'maximum' else signal + patch
            signal = np.minimum(signal, max_contrast)
            drift = rng.uniform(-.6, .6, 3)
            # shortcut: procedural grayscale is not an optical simulation; validate on real frames before deployment.
            raw = reference.astype(np.float32) - signal + drift[0] + drift[1] * nx + drift[2] * ny
            raw += rng.normal(0, lighting['noise_std'], shape)
            raw = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
            diff = reference.astype(np.float32) - raw.astype(np.float32)
            pseudo_height = sensor.raw_image_2_height_map(raw).astype(np.float32)
            save_sample(output, f'synthetic_v2_{seed}_{sample_offset + local_index:06d}', diff, pseudo_height, raw, reference,
                        {'group': f'synthetic_v2_{seed}_contact_{group_index:05d}',
                         'object_id': f'synthetic_v2_{seed}_object_{group_index:05d}',
                         'data_source': 'synthetic', 'label_source': 'synthetic_traditional_pseudolabel',
                         'generator_version': 2, 'seed': seed,
                         'synthetic_shape': 'background' if background else ('composite' if count > 1 else components[0]['family']),
                         'components': components, 'blend_mode': blend, 'lighting': lighting,
                         'frame_transform': {'shift_xy_px': shift.tolist(), 'size_gain': float(size_gain),
                                             'contrast_gain': float(contrast_gain), 'lighting_drift': drift.tolist()},
                         'mm_per_pixel': scale, 'max_height_mm': float(pseudo_height.max()),
                         'is_background': bool(pseudo_height.max() <= .05),
                         'lighting_threshold': sensor.lighting_threshold, 'kernel_list': list(sensor.kernel_list)})
    print(f'Generated {num_samples} new synthetic samples in {output}; total {sample_offset + num_samples}')
    return num_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input_dir', help='Directory with raw_images/ and optional reference_images/, samples.jsonl')
    source.add_argument('--synthetic', action='store_true', help='Randomly synthesize contact/reference images; no input images or camera')
    parser.add_argument('--output_dir', default='reconstruction_data/raw')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument('--reference', help='Shared no-contact image; otherwise pair by filename in reference_images/')
    parser.add_argument('--group', help='Actual contact group for ALL input frames; otherwise read samples.jsonl')
    parser.add_argument('--object_id', help='Actual object identity; otherwise preserve saved metadata')
    parser.add_argument('--num_samples', type=int, default=120, help='Synthetic sample count (default: 120)')
    parser.add_argument('--samples_per_group', type=int, default=5, help='Synthetic frames sharing one contact geometry (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Reproducible synthetic random seed (default: 42)')
    parser.add_argument('--append', action='store_true', help='Append num_samples NEW synthetic frames to an existing complete dataset')
    args = parser.parse_args()
    if args.synthetic:
        if args.reference or args.group or args.object_id:
            parser.error('--synthetic generates its own references, groups and object IDs')
        generate_synthetic(args.output_dir, args.config, args.num_samples, args.samples_per_group, args.seed, args.append)
    else:
        if args.append:
            parser.error('--append requires --synthetic')
        generate(args.input_dir, args.output_dir, args.config, args.reference, args.group, args.object_id)


if __name__ == '__main__':
    main()
