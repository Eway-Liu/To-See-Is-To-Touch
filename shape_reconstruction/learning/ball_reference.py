"""Create a spherical indentation geometry reference from a measured ball radius and contact rim."""
import argparse
from pathlib import Path
import numpy as np


def spherical_cap_reference(shape, center_xy, rim_radius_px, ball_radius_mm, mm_per_pixel):
    if len(shape) != 2 or min(shape) <= 0 or ball_radius_mm <= 0 or mm_per_pixel <= 0:
        raise ValueError('Shape, ball radius, and pixel scale must be positive')
    if not np.isfinite([*center_xy, rim_radius_px, ball_radius_mm, mm_per_pixel]).all():
        raise ValueError('Geometry parameters must be finite')
    if rim_radius_px <= 0 or rim_radius_px * mm_per_pixel >= ball_radius_mm:
        raise ValueError('Contact rim radius must be positive and smaller than the ball radius')
    x,y=center_xy
    if x-rim_radius_px<0 or y-rim_radius_px<0 or x+rim_radius_px>shape[1]-1 or y+rim_radius_px>shape[0]-1:
        raise ValueError('The complete contact circle must lie inside the image')
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    radial_mm = np.hypot(xx - center_xy[0], yy - center_xy[1]) * mm_per_pixel
    rim_mm = rim_radius_px * mm_per_pixel
    rim_sag = np.sqrt(ball_radius_mm ** 2 - rim_mm ** 2)
    height = np.sqrt(np.maximum(ball_radius_mm ** 2 - radial_mm ** 2, 0)) - rim_sag
    height[radial_mm > rim_mm] = 0
    return np.maximum(height, 0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, help='Saved grayscale contact image .npy')
    p.add_argument('--reference', required=True, help='Saved no-contact grayscale reference .npy')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--sample', required=True, help='Output sample stem')
    p.add_argument('--center', nargs=2, type=float, required=True, metavar=('X_PX','Y_PX'))
    p.add_argument('--rim_radius_px', type=float, required=True, help='Manually verified contact-rim radius in pixels')
    p.add_argument('--ball_radius_mm', type=float, required=True, help='Measured ball radius in mm')
    p.add_argument('--mm_per_pixel', type=float, required=True, help='From camera calibration')
    args = p.parse_args()
    raw = np.load(args.raw, allow_pickle=False)
    ref = np.load(args.reference, allow_pickle=False)
    if raw.ndim != 2 or raw.shape != ref.shape or raw.dtype!=np.uint8 or ref.dtype!=np.uint8:
        raise ValueError('Raw and reference images must be same-sized uint8 grayscale arrays')
    label = spherical_cap_reference(raw.shape, args.center, args.rim_radius_px,
                                    args.ball_radius_mm, args.mm_per_pixel)
    out = Path(args.output_dir)
    diff = ref.astype(np.float32)-raw.astype(np.float32)
    from ..data.data_preparation import save_sample
    save_sample(out,args.sample,diff,label,raw,ref,
                {'group':args.sample,'object_id':f'ball_R{args.ball_radius_mm:g}mm',
                 'label_source':'spherical_geometry_reference','center_xy_px':args.center,
                 'rim_radius_px':args.rim_radius_px,'ball_radius_mm':args.ball_radius_mm,
                 'mm_per_pixel':args.mm_per_pixel})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(raw,cmap='gray',vmin=0,vmax=255)
    axes[0].add_patch(Circle(args.center,args.rim_radius_px,fill=False,color='red'))
    axes[0].set_title('Manually verify the red contact rim')
    image=axes[1].imshow(label,cmap='turbo');fig.colorbar(image,ax=axes[1],label='mm')
    axes[1].set_title('Geometry reference, not measured truth')
    fig.tight_layout();fig.savefig(out/f'{args.sample}_alignment.png',dpi=150);plt.close(fig)
    print(f'Wrote {out}; peak cap height={label.max():.3f}mm. Inspect rim alignment before scoring.')


if __name__ == '__main__':
    main()
