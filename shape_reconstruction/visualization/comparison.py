"""Shared physical coordinates and color scale for both reconstruction methods."""
import numpy as np


def height_points(height, mm_per_pixel, offset_x=0.0):
    if height.ndim != 2 or not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0 or not np.isfinite(height).all():
        raise ValueError('Invalid height map or physical pixel scale')
    yy, xx = np.indices(height.shape)
    return np.column_stack((xx.ravel()*mm_per_pixel+offset_x,
                            -yy.ravel()*mm_per_pixel, height.ravel()))


def height_colors(height, display_max_mm):
    t = np.clip(height.ravel()/display_max_mm, 0, 1)
    # One fixed blue-to-red ramp shared across methods and frames.
    return np.column_stack((t, .3+.5*(1-np.abs(2*t-1)), 1-t))


class ComparisonVisualizer:
    def __init__(self, height_maps, mm_per_pixel, display_max_mm=3.0, names=None):
        import open3d as o3d
        self.o3d = o3d
        self.mm_per_pixel, self.display_max_mm = mm_per_pixel, display_max_mm
        self.vis = o3d.visualization.Visualizer()
        title = ' | '.join(names or ['traditional','neural'][:len(height_maps)])
        if not self.vis.create_window(f'3D: {title}, units mm', 1400, 800):
            raise RuntimeError('Open3D window could not be created; use --headless for export')
        self.clouds = []
        self.spacing = height_maps[0].shape[1]*mm_per_pixel+5
        for i, height in enumerate(height_maps):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(height_points(height, mm_per_pixel, i*self.spacing))
            cloud.colors = o3d.utility.Vector3dVector(height_colors(height, display_max_mm))
            self.vis.add_geometry(cloud)
            self.clouds.append(cloud)
        ctr = self.vis.get_view_control()
        ctr.set_front([0,-.7,1]); ctr.set_up([0,1,0]); ctr.set_zoom(.7)
        self.vis.get_render_option().point_size = 2

    def update(self, maps):
        for i, (cloud, height) in enumerate(zip(self.clouds, maps)):
            cloud.points = self.o3d.utility.Vector3dVector(height_points(height,self.mm_per_pixel,i*self.spacing))
            cloud.colors = self.o3d.utility.Vector3dVector(height_colors(height,self.display_max_mm))
            self.vis.update_geometry(cloud)
        opened = self.vis.poll_events()
        self.vis.update_renderer()
        return opened

    def close(self):
        self.vis.destroy_window()


def export_surfaces(path, maps, names, mm_per_pixel, display_max_mm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7*len(maps),5))
    for i,(height,name) in enumerate(zip(maps,names)):
        ax = fig.add_subplot(1,len(maps),i+1,projection='3d')
        yy,xx = np.indices(height.shape)
        step=max(1,max(height.shape)//120)
        ax.plot_surface(xx[::step,::step]*mm_per_pixel,-yy[::step,::step]*mm_per_pixel,
                        height[::step,::step],cmap='turbo',vmin=0,vmax=display_max_mm,linewidth=0)
        ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='height (mm)',title=name,zlim=(0,display_max_mm))
        ax.set_box_aspect((height.shape[1]*mm_per_pixel,height.shape[0]*mm_per_pixel,display_max_mm))
        ax.view_init(elev=35,azim=-65)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def write_ply(path, height, mm_per_pixel):
    points = height_points(height, mm_per_pixel)
    with open(path,'w',encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f,points,fmt='%.6f')
