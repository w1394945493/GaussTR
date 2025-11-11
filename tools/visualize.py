import argparse
import os
# os.environ['ETS_TOOLKIT'] = 'null'
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 强制 Qt 使用 offscreen 渲染

import pickle
from glob import glob

import numpy as np
from mayavi import mlab
# from rich.progress import track
from tqdm import tqdm

# COLORS = np.array([
#     [0, 0, 0, 255],
#     [112, 128, 144, 255],
#     [220, 20, 60, 255],
#     [255, 127, 80, 255],
#     [255, 158, 0, 255],
#     [233, 150, 70, 255],
#     [255, 61, 99, 255],
#     [0, 0, 230, 255],
#     [47, 79, 79, 255],
#     [255, 140, 0, 255],
#     [255, 98, 70, 255],
#     [0, 207, 191, 255],
#     [175, 0, 75, 255],
#     [75, 0, 75, 255],
#     [112, 180, 60, 255],
#     [222, 184, 135, 255],
#     [0, 175, 0, 255],
# ])

COLORS = np.array(
    [
        [  0,   0,   0, 255],       # others
        [255, 120,  50, 255],       # barrier              orange
        [255, 192, 203, 255],       # bicycle              pink
        [255, 255,   0, 255],       # bus                  yellow
        [  0, 150, 245, 255],       # car                  blue
        [  0, 255, 255, 255],       # construction_vehicle cyan
        [255, 127,   0, 255],       # motorcycle           dark orange
        [255,   0,   0, 255],       # pedestrian           red
        [255, 240, 150, 255],       # traffic_cone         light yellow
        [135,  60,   0, 255],       # trailer              brown
        [160,  32, 240, 255],       # truck                purple
        [255,   0, 255, 255],       # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [ 75,   0,  75, 255],       # sidewalk             dard purple
        [150, 240,  80, 255],       # terrain              light green
        [230, 230, 250, 255],       # manmade              white
        [  0, 175,   0, 255],       # vegetation           green
        # [  0, 255, 127, 255],       # ego car              dark cyan
        # [255,  99,  71, 255],       # ego car
        # [  0, 191, 255, 255]        # ego car
    ]
).astype(np.uint8)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


def get_grid_coords(grid_shape, voxel_size):
    coords = np.meshgrid(*[np.arange(0, s) for s in grid_shape])
    coords = np.array([i.flatten() for i in coords]).T.astype(float)
    coords = coords * voxel_size + voxel_size / 2
    coords = np.stack([coords[:, 1], coords[:, 0], coords[:, 2]], axis=1)
    return coords


def plot(
        voxels,
        colors,
        voxel_size=0.4,
        ignore_labels=(17, 255),
        bg_color=(1, 1, 1),
        save=False,
):
    voxels = np.vstack(
        [get_grid_coords(voxels.shape, voxel_size).T,
         voxels.flatten()]).T
    for lbl in ignore_labels:
        voxels = voxels[voxels[:, 3] != lbl]
    # 禁用后端显示
    mlab.options.off_screen = True  # 这行是禁用GUI后端的关键

    mlab.figure(bgcolor=bg_color)


    # plt_plot = mlab.points3d(
    #     *voxels.T,
    #     scale_factor=voxel_size,
    #     mode='cube',
    #     opacity=1.0,
    #     vmin=0,
    #     vmax=16)
    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    scalars = voxels[:, 3]

    plt_plot = mlab.points3d(
        x, y, z,
        scalars,
        scale_factor=voxel_size,
        mode='cube',
        opacity=1.0,
        vmin=0,
        vmax=16,
    )
    plt_plot.glyph.scale_mode = 'scale_by_vector'
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    plt_plot.scene.camera.zoom(1.2)
    if save:
        mlab.savefig(save, size=(1200, 1200))
        mlab.close()



def main():
    args = parse_args()
    files = glob(f'{args.path}/*.pkl')
    save_dir = '/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/test/visualizations'
    os.makedirs(save_dir,exist_ok=True)


    # for file in track(files):
    for file in tqdm(files):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)

        file_name = file.split(os.sep)[-1].split('.')[0]
        for i, occ in enumerate((outputs['occ_pred'], outputs['occ_gt'])):
            plot(
                occ,
                colors=COLORS,
                # save=f"visualizations/{file_name}_{'gt' if i else 'pred'}.png"
                save=f"{save_dir}/{file_name}_{'gt' if i else 'pred'}.png"
                if args.save else None)


if __name__ == '__main__':
    main()
