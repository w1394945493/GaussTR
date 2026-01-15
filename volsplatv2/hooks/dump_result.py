import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import pickle

from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

from pyquaternion import Quaternion
from tqdm import tqdm

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"



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
).astype(np.float32) / 255.



@HOOKS.register_module()
class DumpResultHook(Hook):
    def __init__(self,
                 save_dir='output/vis',    
                 save_occ=True,  
                 save_gaussian=True,           
                 ):
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir    
        self.save_occ = save_occ
        self.save_gaussian = save_gaussian
        
        if save_occ:
            self.occ_depth = os.path.join(save_dir, 'occ_pred')
            os.makedirs(self.occ_depth,exist_ok=True)            
        print(f"Dump results to: {self.save_dir}")        
        return
    
    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        
        bs, n = data_batch['cam2img'].shape[:2]   

        if self.save_occ and ('occ_pred' in outputs[0]):
             for i in range(bs):
                # todo --------------------------------------#
                # todo 保存占用图
                occ_pred = outputs[0]['occ_pred'][i].cpu().numpy()
                occ_gt = outputs[0]['occ_gt'][i].cpu().numpy()
                output = dict(
                    occ_pred = occ_pred,
                    occ_gt = occ_gt, )                    
                save_name = f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}.pkl"
                save_path = os.path.join(self.occ_depth, save_name)
                with open(save_path, 'wb') as f:
                    pickle.dump(output, f)  
                
                # todo --------------------------------------#
                # todo 可视化高斯分布
                # if 'gaussian' in outputs[0] and self.save_gaussian:
                #     means = outputs[0]['gaussian'].means[i].cpu().numpy()
                #     scales = outputs[0]['gaussian'].scales[i].cpu().numpy()
                #     rotations = outputs[0]['gaussian'].rotations[i].cpu().numpy()
                #     opas = outputs[0]['gaussian'].opacities[i].cpu().numpy()
                #     sems = outputs[0]['gaussian'].semantics[i].cpu().numpy() # (n,18)
                    
                #     save_gaussian(
                #         self.occ_depth,
                #         means, scales, rotations, opas, sems,
                #         f"{data_batch['scene_token'][i]}_{data_batch['token'][i]}",)
                                
        return

def save_gaussian(save_dir, means, scales, rotations, opas, sems,
                  name, scalar=1.5, ignore_opa=False, filter_zsize=False):
    empty_label = 17
    sem_cmap = COLORS
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.75)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk

        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)
    
    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:],
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in tqdm(range(ellipNumber)):

        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T # todo 四元数转旋转矩阵

        # calculate cartesian coordinates for the ellipsoid surface
        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2],
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    # plt.gca().set_box_aspect([1, 1, 1])
    ax.grid(False)
    ax.set_axis_off()

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)

    plt.cla()
    plt.clf()    