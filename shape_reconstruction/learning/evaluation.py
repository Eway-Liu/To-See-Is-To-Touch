"""Held-out evaluation in mm; pseudo-label agreement is explicitly labeled."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from ..data.pairs import load_pair, manifest_root


def error_metrics(pred, target, threshold=.05):
    error = pred.astype(np.float64) - target
    contact = target > threshold
    predicted = pred > threshold
    union = np.count_nonzero(contact | predicted)
    edge = sum(np.abs(np.diff(pred, axis=a)-np.diff(target, axis=a)).mean() for a in (0,1)) / 2
    return {'mae_mm': float(np.abs(error).mean()),
            'rmse_mm': float(np.sqrt(np.square(error).mean())),
            'contact_mae_mm': float(np.abs(error[contact]).mean()) if contact.any() else None,
            'background_mae_mm': float(np.abs(error[~contact]).mean()) if (~contact).any() else None,
            'contact_iou': float(np.count_nonzero(contact & predicted)/union) if union else 1.0,
            'gradient_mae_mm_per_pixel': float(edge), 'max_height_error_mm': float(abs(pred.max()-target.max()))}


def traditional_from_saved(root, name, config=None):
    """Call the unmodified traditional algorithm with saved raw/reference frames."""
    from ..core.sensor import Sensor
    from ..config import load_config
    cfg = load_config(config) if config else load_config()
    reference = np.load(Path(root)/'reference_images'/name, allow_pickle=False)
    raw = np.load(Path(root)/'raw_images'/name, allow_pickle=False)
    sensor = Sensor.from_reference(cfg, reference)
    if raw.shape != reference.shape or raw.dtype != np.uint8:
        raise ValueError('Saved raw/reference frames must be matching uint8 grayscale arrays')
    return Sensor.raw_image_2_height_map(sensor, raw)


def evaluate(model, root, names, output, label_source='unknown', config=None, compare_traditional=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not names:
        raise ValueError('Empty evaluation set')
    rows, all_metrics = [], []
    model.predict(load_pair(root, names[0], model.height_scale)[0])  # warm-up excluded
    for index, name in enumerate(names):
        diff, target = load_pair(root, name, model.height_scale)
        start = time.perf_counter()
        pred = model.predict(diff)  # includes transfer back to CPU, hence CUDA completion
        elapsed = (time.perf_counter()-start)*1000
        item = error_metrics(pred, target)
        row = {'file':name, 'method':'neural_network', 'inference_ms':elapsed, **item}
        rows.append(row)
        all_metrics.append(item)
        traditional = None
        if compare_traditional:
            traditional = traditional_from_saved(root, name, config)
            # Timing includes loading/calibration: do not compare it to model inference.
            rows.append({'file':name,'method':'traditional', 'inference_ms':None, **error_metrics(traditional,target)})
        if index < 8:
            save_panel(output/f'sample_{index:02d}.png', diff, target, pred, traditional)
    keys = list(all_metrics[0])
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method']==method]
        summary[method] = {k: float(np.mean([r[k] for r in selected if r[k] is not None]))
                           if any(r[k] is not None for r in selected) else None for k in keys}
    times = [r['inference_ms'] for r in rows if r['inference_ms'] is not None]
    report = {'samples':len(names), 'label_source':label_source,
              'interpretation':'Agreement with supplied reference. Pseudo-label metrics do not establish physical accuracy.',
              'aggregation':'Mean per-frame metrics; RMSE is mean per-frame RMSE.', 'metrics':summary,
              'neural_latency_ms': {'median':float(np.median(times)), 'p95':float(np.percentile(times,95))}}
    (output/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),encoding='utf-8')
    with (output/'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def save_panel(path, diff, target, pred, traditional=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames = [('Input difference',diff)]
    if traditional is not None: frames.append(('Traditional (mm)',traditional))
    frames += [('Reference (mm)',target), ('Neural (mm)',pred), ('Abs. error (mm)',np.abs(pred-target))]
    fig, axes = plt.subplots(1,len(frames),figsize=(4*len(frames),3.5))
    vmax = max(float(target.max()),float(pred.max()),float(traditional.max()) if traditional is not None else 0,.1)
    for i,(title,array) in enumerate(frames):
        image = axes[i].imshow(array,cmap='coolwarm' if i==0 else 'turbo',vmin=-255 if i==0 else 0, vmax=255 if i==0 else vmax)
        axes[i].set_title(title); axes[i].axis('off'); fig.colorbar(image,ax=axes[i],fraction=.04)
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path',required=True)
    p.add_argument('--data_dir')
    p.add_argument('--manifest')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',default='comparison_results')
    p.add_argument('--device')
    p.add_argument('--config')
    p.add_argument('--compare_traditional',action='store_true')
    p.add_argument('--label_source', default='unknown')
    args=p.parse_args()
    if args.manifest:
        root,doc=manifest_root(args.manifest,args.data_dir)
        names,source=doc['splits'][args.split],doc['label_source']
    else:
        if not args.data_dir: p.error('Provide --manifest or --data_dir')
        root=args.data_dir; names=sorted(p.name for p in (Path(root)/'diff_images').glob('*.npy')); source=args.label_source
    from .inference import ReconstructionModel
    evaluate(ReconstructionModel(args.model_path,args.device),root,names,args.output,source,args.config,args.compare_traditional)


if __name__ == '__main__':
    main()
