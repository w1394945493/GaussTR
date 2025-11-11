# Refered to https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_local_demo.py
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert
from tqdm import tqdm

# import hydra
# from omegaconf import DictConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

# from torch.cuda.amp import autocast
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
        [  0,   0,   0, 255],       # others               black 黑色
        [255, 120,  50, 255],       # barrier              orange 橙色
        [255, 192, 203, 255],       # bicycle              pink 粉色
        [255, 255,   0, 255],       # bus                  yellow 黄色
        [  0, 150, 245, 255],       # car                  blue 蓝色
        [  0, 255, 255, 255],       # construction_vehicle cyan 青色
        [255, 127,   0, 255],       # motorcycle           dark orange 深橙色
        [255,   0,   0, 255],       # pedestrian           red 红色
        [255, 240, 150, 255],       # traffic_cone         light yellow 浅黄色
        [135,  60,   0, 255],       # trailer              brown 棕色
        [160,  32, 240, 255],       # truck                purple 紫色
        [255,   0, 255, 255],       # driveable_surface    dark pink 深粉色
        # [175,   0,  75, 255],     # other_flat           dark red 深红色
        [139, 137, 137, 255],       # 无特定分类            gray 灰色
        [ 75,   0,  75, 255],       # sidewalk             dark purple 深紫色
        [150, 240,  80, 255],       # terrain              light green 浅绿色
        [230, 230, 250, 255],       # manmade              white 白色
        [  0, 175,   0, 255],       # vegetation           green 绿色
        # [  0, 255, 127, 255],     # ego car              dark cyan 深青色
        # [255,  99,  71, 255],     # ego car              red 红色
        # [  0, 191, 255, 255]      # ego car              light blue 浅蓝色
    ]
)

OCC3D_CATEGORIES = (
    ['barrier', 'concrete barrier', 'metal barrier', 'water barrier'],
    ['bicycle', 'bicyclist'],
    ['bus'],
    ['car'],
    ['crane'],
    ['motorcycle', 'motorcyclist'],
    ['pedestrian', 'adult', 'child'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['traffic island', 'rail track', 'lake', 'river'],
    ['sidewalk'],
    ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage',
     'hydrant', 'street sign', 'traffic light'],
    ['tree', 'bush'],
    ['sky', 'empty'],
)
CLASSES = sum(OCC3D_CATEGORIES, [])
TEXT_PROMPT = '. '.join(CLASSES)
INDEX_MAPPING = [
    outer_index for outer_index, inner_list in enumerate(OCC3D_CATEGORIES)
    for _ in inner_list
]

# IMG_PATH = 'data/nuscenes/samples/'
# OUTPUT_DIR = Path('nuscenes_grounded_sam2/')

IMG_PATH = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini/samples/'
OUTPUT_DIR = Path('/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_grounded_sam2/mini')
VIS_DIR = Path('/home/lianghao/wangyushen/data/wangyushen/Output/gausstr/sam2/seg_vis/mini')
# SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
# GROUNDING_DINO_CONFIG = 'grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py'

SAM2_MODEL_CONFIG = '/home/lianghao/wangyushen/Projects/GaussTR/configs/sam2.1/sam2.1_hiera_b+.yaml'
GROUNDING_DINO_CONFIG = '/home/lianghao/wangyushen/Projects/GaussTR/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py'

# SAM2_CHECKPOINT = 'checkpoints/sam2.1_hiera_base_plus.pt'
# GROUNDING_DINO_CHECKPOINT = 'gdino_checkpoints/groundingdino_swinb_cogcoor.pth'
SAM2_CHECKPOINT = '/home/lianghao/wangyushen/data/wangyushen/Weights/grounding_dino_sam2/checkpoints/sam2.1_hiera_base_plus.pt'
GROUNDING_DINO_CHECKPOINT = '/home/lianghao/wangyushen/data/wangyushen/Weights/grounding_dino_sam2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth'


BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DUMP_JSON_RESULTS = True



VIEW_DIRS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
    'CAM_BACK_LEFT',
]

def main():
    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True,exist_ok=True)

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG


    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)



    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE)

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT

    i_iter = 0
    vis = False
    # for view_dir in os.listdir(IMG_PATH):
    for view_dir in tqdm(VIEW_DIRS):
        for image_path in tqdm(os.listdir(osp.join(IMG_PATH, view_dir))): # image_path:图片名
            image_source, image = load_image(
                os.path.join(IMG_PATH, view_dir, image_path))

            sam2_predictor.set_image(image_source)

            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(
                boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            # FIXME: figure how does this influence the G-DINO model
            # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            if input_boxes.shape[0] != 0:
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                # convert the shape to (n, H, W)
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

            results = np.zeros_like(masks[0])
            if input_boxes.shape[0] != 0:
                for i in range(len(labels)):
                    if labels[i] not in CLASSES:
                        continue
                    pred = INDEX_MAPPING[CLASSES.index(labels[i])] + 1
                    results[masks[i].astype(bool)] = pred

            i_iter += 1
            # if vis and (i_iter % 10 == 0):
            if vis:
                height, width = results.shape
                color_image = np.zeros((height, width, 4), dtype=np.uint8)

                # 遍历每个像素，给它分配对应类别的颜色
                for i in range(height):
                    for j in range(width):
                        category_index = int(results[i, j])  # 获取类别索引
                        if category_index < len(COLORS):  # 确保索引不超出颜色范围
                            color_image[i, j] = COLORS[category_index]
                image_name = image_path.split('.')[0]
                save_path = os.path.join(VIS_DIR,image_name)
                plt.imsave(f'{save_path}_seg.png', color_image)
                ori_image_path = os.path.join(IMG_PATH, view_dir, image_path)
                ori_image = Image.open(ori_image_path)
                ori_save_path = f'{save_path}.png'
                ori_image.save(ori_save_path)


            np.save(osp.join(OUTPUT_DIR, image_path.split('.')[0]), results)


if __name__=='__main__':
    main()