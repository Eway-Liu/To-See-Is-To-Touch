"""Read-only full dataset inspection, including missing, malformed and empty pairs."""
import argparse
from pathlib import Path
import numpy as np
from .pairs import load_pair


def analyze_dataset(data_dir='reconstruction_data/raw',height_scale=10.):
    root=Path(data_dir); rows=[]; invalid=[]
    files=sorted((root/'diff_images').glob('*.npy'))
    for file in files:
        try:
            diff,height=load_pair(root,file.name,height_scale)
        except (ValueError,OSError,EOFError) as exc:
            invalid.append((file.name,str(exc))); continue
        mask=height>.05
        rows.append((file.name,diff,height,float(height.max()),float(height[mask].mean()) if mask.any() else 0.,int(mask.sum())))
    print(f'Pairs: {len(rows)} valid / {len(files)} inputs; invalid: {len(invalid)}')
    for name,reason in invalid: print(f'INVALID {name}: {reason}')
    orphan=sorted(p.name for p in (root/'height_maps').glob('*.npy') if not (root/'diff_images'/p.name).is_file())
    print(f'Orphan height maps: {len(orphan)}; background samples: {sum(r[5]==0 for r in rows)}')
    if rows:
        peaks=np.array([r[3] for r in rows]); areas=np.array([r[5] for r in rows])
        print(f'Peak height p05/median/p95 mm: {np.percentile(peaks,[5,50,95])}')
        print(f'Contact area min/median/max pixels: {areas.min()}/{np.median(areas)}/{areas.max()}')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',default='reconstruction_data/raw')
    p.add_argument('--height_scale',type=float,default=10.)
    p.add_argument('--visualize',action='store_true')
    p.add_argument('--output',default='dataset_samples.png')
    a=p.parse_args();rows=analyze_dataset(a.data_dir,a.height_scale)
    if a.visualize and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected=np.random.default_rng(42).choice(len(rows),min(9,len(rows)),replace=False)
        fig,axes=plt.subplots(len(selected),2,figsize=(9,3*len(selected)),squeeze=False)
        vmax=max(rows[i][3] for i in selected) or .1
        for ax,i in zip(axes,selected):
            name,diff,height,*_=rows[i]
            ax[0].imshow(diff,cmap='gray',vmin=0,vmax=255);ax[0].set_title(name)
            image=ax[1].imshow(height,cmap='turbo',vmin=0,vmax=vmax);ax[1].set_title('Reference height (mm)')
            fig.colorbar(image,ax=ax[1])
        fig.tight_layout();fig.savefig(a.output,dpi=120);plt.close(fig)
        print(f'Saved {a.output}')


if __name__=='__main__':main()
