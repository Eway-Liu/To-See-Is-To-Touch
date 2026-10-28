"""Train a dense height regressor; metrics use mm and checkpoint stores preprocessing."""
import argparse
import hashlib
import json
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import ShapeReconstructionDataset
from .model import build_model, count_parameters
from ..data.pairs import manifest_root


def atomic_save(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_loaders(args):
    if args.manifest:
        root, document = manifest_root(args.manifest, args.data_dir)
        scale = document.get('height_scale_mm', args.height_scale)
        for left, right in (('train','val'), ('train','test'), ('val','test')):
            if set(document['splits'][left]) & set(document['splits'][right]):
                raise ValueError(f'Overlapping samples between {left} and {right}')
            rows = {r['file']:r for r in document['samples']}
            for key in ('group','input_sha256'):
                a = {rows[n][key] for n in document['splits'][left] if rows[n].get(key)}
                b = {rows[n][key] for n in document['splits'][right] if rows[n].get(key)}
                if a & b:
                    raise ValueError(f'Overlapping {key} between {left} and {right}')
        train = ShapeReconstructionDataset('train', root, manifest=args.manifest,
                                           height_scale=scale, augmentation=True, geometric=args.geometric,
                                           input_mode=args.input_mode)
        val = ShapeReconstructionDataset('val', root, manifest=args.manifest,
                                         height_scale=scale, augmentation=False, input_mode=args.input_mode)
        if not document['splits']['test']:
            print('WARNING: no independent test split; validation metrics do not establish generalization.')
        return train, val, scale
    if not args.train_dir:
        args.train_dir = str(Path(args.data_dir or './reconstruction_data') / 'train')
        args.val_dir = str(Path(args.data_dir or './reconstruction_data') / 'val')
    train = ShapeReconstructionDataset('train', args.train_dir, augmentation=True,
                                       height_scale=args.height_scale, geometric=args.geometric,
                                       input_mode=args.input_mode)
    if not args.val_dir or Path(args.train_dir).resolve() == Path(args.val_dir).resolve():
        raise ValueError('Provide an independent --val_dir or use --manifest')
    val = ShapeReconstructionDataset('val', args.val_dir or args.train_dir, augmentation=False,
                                     height_scale=args.height_scale, input_mode=args.input_mode)
    return train, val, args.height_scale


def metrics(pred, target, scale):
    error = (pred.float() - target.float()) * scale
    return {'mae_mm': error.abs().mean().item(), 'rmse_mm': error.square().mean().sqrt().item()}


def loss_fn(pred, target, scale=10.0):
    p, t = pred.float(), target.float()
    # Emphasize loaded/contact pixels while preserving empty background evidence.
    weight = 1.0 + 3.0 * (t > .05 / scale).float()
    data_loss = (nn.functional.smooth_l1_loss(p, t, reduction='none', beta=.02) * weight).mean()
    dx = nn.functional.l1_loss(p[..., 1:] - p[..., :-1], t[..., 1:] - t[..., :-1])
    dy = nn.functional.l1_loss(p[..., 1:, :] - p[..., :-1, :], t[..., 1:, :] - t[..., :-1, :])
    return data_loss + .15 * (dx + dy)


def run_epoch(model, loader, optimizer, scaler, device, amp, train, scale=10.0,
              *, epoch=None, epochs=None, progress='off'):
    model.train(train)
    total, mae, rmse, count = 0.0, 0.0, 0.0, 0
    phase = 'Train' if train else 'Valid'
    description = f'{phase} [{epoch}/{epochs}]' if epoch is not None else phase
    show_bar = progress == 'on' or (progress == 'auto' and sys.stdout.isatty())
    show_text = progress == 'auto' and not show_bar
    batch_count = len(loader)
    started = last_report = time.perf_counter()
    if show_text:
        print(f'  {description} 0/{batch_count} batches | loading first batch / warm-up...', flush=True)
    context = torch.enable_grad if train else torch.inference_mode
    with context(), tqdm(loader, desc=description, unit='batch', file=sys.stdout,
                         dynamic_ncols=True, ascii=True, leave=False,
                         bar_format='{desc} {n_fmt}/{total_fmt}{postfix} |{bar}| ETA {remaining} {rate_fmt}',
                         mininterval=.5, miniters=1, disable=not show_bar) as batches:
        for step, (x, y) in enumerate(batches, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == 'cuda'):
                pred = model(x)
                loss = loss_fn(pred, y, scale)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; check data and learning rate')
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            batch = x.shape[0]
            m = metrics(pred.detach(), y, scale)
            total += loss.item() * batch
            mae += m['mae_mm'] * batch
            rmse += m['rmse_mm'] ** 2 * batch
            count += batch
            if show_bar:
                values = {'loss': f'{total/count:.5f}', 'MAE': f'{mae/count:.4f}mm'}
                batches.set_postfix(values, refresh=False)
            elif show_text:
                now = time.perf_counter()
                if step == 1 or step == batch_count or now - last_report >= 10:
                    elapsed = now - started
                    eta = elapsed / step * (batch_count - step)
                    print(f'  {description} {step}/{batch_count} batches '
                          f'({step/batch_count:.0%}) | loss={total/count:.5f} '
                          f'MAE={mae/count:.4f}mm | '
                          f'{step/max(elapsed, 1e-9):.2f} batch/s | '
                          f'elapsed={timedelta(seconds=int(elapsed))} '
                          f'ETA={timedelta(seconds=int(eta))}', flush=True)
                    last_report = now
    if not count:
        return None
    return {'loss': total/count, 'mae_mm': mae/count, 'rmse_mm': (rmse/count) ** .5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', help='Data root; overrides manifest path after copying to another computer')
    p.add_argument('--manifest')
    p.add_argument('--train_dir')
    p.add_argument('--val_dir')
    p.add_argument('--model', choices=['attention_unet','unet','simple','resnet'], default='attention_unet')
    p.add_argument('--base_features', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--height_scale', type=float, default=10.0)
    p.add_argument('--save_dir', default='./saved_models')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=18)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--geometric', action='store_true', help='Enable flips/180-degree turns only after validation')
    p.add_argument('--input_mode', choices=['positive_diff','signed_diff'], default='positive_diff')
    p.add_argument('--progress', choices=['auto', 'on', 'off'], default='auto',
                   help='auto: terminal bars or periodic plain logs; on: force bars; off: epoch summaries only')
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.base_features < 4 or args.workers < 0 or args.patience < 1:
        raise ValueError('Invalid training parameters')
    if not np.isfinite([args.height_scale,args.lr]).all() or args.height_scale<=0 or args.lr<=0:
        raise ValueError('Height scale and learning rate must be finite and positive')
    out = Path(args.save_dir).resolve()
    history_path = out / 'history.jsonl'
    if (history_path.exists() or (out/'latest.pth').exists()) and not args.resume:
        raise ValueError('Save directory already has training results; use a new directory or --resume')
    if args.resume and (Path(args.resume).resolve().parent!=out or not (out/'reconstruction_best.pth').is_file()):
        raise ValueError('Resume in the original --save_dir, keeping reconstruction_best.pth alongside latest.pth')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a CUDA-enabled PyTorch build.')
    print('Preparing training and validation data...', flush=True)
    train_ds, val_ds, scale = make_loaders(args)
    if train_ds.shape!=val_ds.shape:
        raise ValueError('Training and validation image dimensions differ')
    fingerprint = None
    if args.manifest:
        _, document = manifest_root(args.manifest, args.data_dir)
        document.pop('data_dir',None)
        fingerprint = hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest()
    kwargs = {'num_workers': args.workers, 'pin_memory': device.type == 'cuda'}
    if args.workers:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **kwargs) if val_ds else []
    model = build_model(args.model, args.base_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=.5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not args.no_amp)
    start, best, stale = 0, float('inf'), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if ckpt.get('model_type') != args.model or ckpt.get('base_features') != args.base_features:
            raise ValueError('Resume checkpoint architecture does not match requested model')
        if ckpt.get('height_scale', 10.0) != scale or ckpt.get('input_mode', 'positive_diff') != args.input_mode:
            raise ValueError('Resume checkpoint normalization does not match dataset')
        if ckpt.get('manifest_sha256') != fingerprint:
            raise ValueError('Resume manifest differs from the original training manifest')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start, best, stale = ckpt['epoch'] + 1, ckpt.get('best_val_loss', best), ckpt.get('stale_epochs', 0)
        if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
    out.mkdir(parents=True, exist_ok=True)
    run = {'device': str(device), 'model': args.model, 'base_features': args.base_features,
           'parameters': count_parameters(model), 'train_samples': len(train_ds),
           'val_samples': len(val_ds) if val_ds else 0, 'height_scale_mm': scale,
           'label_source': getattr(train_ds, 'label_source', 'legacy'), 'input_mode': args.input_mode,
           'manifest_sha256':fingerprint, 'arguments':vars(args)}
    (out/'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print('\n' + '=' * 72, flush=True)
    print(f'Shape reconstruction training | {args.model} | {run["parameters"]:,} parameters', flush=True)
    print(f'Device: {device} ({device_name}) | AMP: {"on" if scaler.is_enabled() else "off"}', flush=True)
    print(f'Data: {len(train_ds):,} train / {len(val_ds):,} validation | '
          f'{train_ds.shape[0]}x{train_ds.shape[1]} | batch size {args.batch_size}', flush=True)
    print(f'Epochs completed: {start}/{args.epochs} | initial lr: {optimizer.param_groups[0]["lr"]:.2g}', flush=True)
    if args.resume:
        print(f'Resumed checkpoint: {Path(args.resume).resolve()}', flush=True)
    print(f'Checkpoints and logs: {out}', flush=True)
    print('=' * 72, flush=True)
    run_started = time.perf_counter()
    for epoch in range(start, args.epochs):
        t0 = time.perf_counter()
        best_text = f'{best:.5f}' if np.isfinite(best) else '--'
        print(f'\nEpoch {epoch+1}/{args.epochs} | lr={optimizer.param_groups[0]["lr"]:.2g} | '
              f'best val loss={best_text} | early stop={stale}/{args.patience}', flush=True)
        tr = run_epoch(model, train_loader, optimizer, scaler, device, not args.no_amp, True, scale,
                       epoch=epoch+1, epochs=args.epochs, progress=args.progress)
        va = run_epoch(model, val_loader, optimizer, scaler, device, not args.no_amp, False, scale,
                       epoch=epoch+1, epochs=args.epochs, progress=args.progress) if len(val_ds) else tr
        scheduler.step(va['loss'])
        improved = va['loss'] < best
        if improved:
            best, stale = va['loss'], 0
        else:
            stale += 1
        ckpt = {'epoch': epoch, 'model_type': args.model, 'base_features': args.base_features,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best, 'stale_epochs': stale, 'height_scale': scale,
                'input_mode': args.input_mode, 'label_source': run['label_source'],
                'manifest_sha256':fingerprint}
        atomic_save(ckpt, out/'latest.pth')
        if improved:
            atomic_save(ckpt, out/'reconstruction_best.pth')
        print(f"  Train: loss={tr['loss']:.5f} | MAE={tr['mae_mm']:.4f}mm", flush=True)
        print(f"  Valid: loss={va['loss']:.5f} | MAE={va['mae_mm']:.4f}mm | "
              f"RMSE={va['rmse_mm']:.4f}mm", flush=True)
        print(f'  Best val loss={best:.5f}' + (' [NEW BEST - saved]' if improved else '') +
              f' | early stop={stale}/{args.patience}', flush=True)
        print(f'  Time: epoch={timedelta(seconds=int(time.perf_counter()-t0))} | '
              f'total time={timedelta(seconds=int(time.perf_counter()-run_started))}', flush=True)
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch+1, 'train': tr, 'val': va, 'seconds': time.perf_counter()-t0}) + '\n')
        if stale >= args.patience:
            print(f'Early stop: no validation loss improvement for {stale} epochs', flush=True)
            break
    print(f'\nBest checkpoint: {out / "reconstruction_best.pth"}', flush=True)
    if 'pseudolabel' in str(run['label_source']):
        print('Metrics describe agreement with traditional pseudo-labels, not physical accuracy.')


if __name__ == '__main__':
    main()
