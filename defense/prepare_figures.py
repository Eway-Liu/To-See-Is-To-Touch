"""Prepare cited source crops and plots from existing project records."""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'images'
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 14,
                     'axes.unicode_minus': False, 'savefig.facecolor': 'white'})

attachments = {
    'pipeline_source': '52c6bc18-6085-4816-83d2-54977d49c739',
    'stability_source': 'ebd7bc4d-8848-4a4d-967b-9c37564e1f8d',
    'deformation_source': '170a2b1e-b044-409b-9b1c-98a7d6a5c10f',
}
for name, ident in attachments.items():
    source = Path('C:/Users/Eway/AppData/Local/Temp') / f'codex-clipboard-{ident}.png'
    target = OUT / f'{name}.png'
    if not target.exists():
        shutil.copy2(source, target)

im = Image.open(OUT / 'pipeline_source.png')
im.crop((round(im.width*.300), 0, round(im.width*.820), round(im.height*.390))).save(OUT / 'reconstruction_flow.png')

record = REPO / 'demo_results/frame_1793155445090073300.npz'
with np.load(record, allow_pickle=False) as data:
    raw = data['raw_gray'].copy()
    traditional = data['traditional'].copy()
    neural = data['neural'].copy()
    scale = float(data['mm_per_pixel'])
assert raw.shape == traditional.shape == neural.shape == (300, 400)
assert all(np.isfinite(x).all() for x in (traditional, neural)) and scale > 0
extent = (0, raw.shape[1]*scale, raw.shape[0]*scale, 0)
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), layout='constrained')
for ax, values, name in zip(axes, [raw, traditional, neural], ['保存的接触图像', '查表重建', '网络重建']):
    if ax is axes[0]:
        ax.imshow(values, cmap='gray', vmin=0, vmax=255, extent=extent)
    else:
        image = ax.imshow(values, cmap='turbo', vmin=0, vmax=.5, extent=extent)
    ax.set_title(name, fontsize=20, pad=12)
    ax.set_xlabel('x / mm')
    ax.set_ylabel('y / mm')
fig.colorbar(image, ax=list(axes[1:]), label='压入深度 / mm', shrink=.82, pad=.025)
fig.savefig(OUT / 'recorded_comparison.png', dpi=190)
plt.close(fig)

fig = plt.figure(figsize=(7.5, 4.7), layout='constrained')
ax = fig.add_subplot(111, projection='3d')
yy, xx = np.indices(traditional.shape)
ax.plot_surface(xx[::4, ::4]*scale, yy[::4, ::4]*scale, traditional[::4, ::4],
                cmap='turbo', vmin=0, vmax=.5, linewidth=0, antialiased=True)
ax.set(xlabel='x / mm', ylabel='y / mm', zlabel='压入深度 / mm', zlim=(0,.5))
ax.set_box_aspect((1.4, 1, .45))
ax.view_init(35, -60)
fig.savefig(OUT / 'recorded_surface.png', dpi=180)
plt.close(fig)

history = [json.loads(line) for line in (REPO / 'saved_models/attention32_final/history.jsonl').read_text().splitlines()]
assert len(history) == 120 and history[-1]['epoch'] == 120
fig, ax = plt.subplots(figsize=(7.2, 3.6), layout='constrained')
for key, name, color in [('train','训练集','#087f8c'), ('val','验证集','#e58c38')]:
    ax.semilogy([r['epoch'] for r in history], [r[key]['mae_mm'] for r in history], label=name, color=color)
ax.set(xlabel='训练轮次', ylabel='相对伪标签的 MAE / mm')
ax.grid(alpha=.18); ax.legend(frameon=False)
fig.savefig(OUT / 'learning_curve.png', dpi=180)
plt.close(fig)
print('Prepared cited crops and 3 evidence plots; assertions passed.')
