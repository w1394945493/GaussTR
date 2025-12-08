import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
# from mmseg.apis import init_segmentor, inference_segmentor

from mmseg.apis import inference_model, init_model
import dinov2.eval.segmentation.models




class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model):
    # model = init_segmentor(cfg)
    model = init_model(cfg)

    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model



# todo 学习：DINOV2提供的 backbone + 分割头 进行语义分割 的 示例
if __name__=='__main__':

    # todo ----------------------------------#
    # todo load pretrained backbone
    BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")


    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    # backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)


    from dinov2.models.vision_transformer import vit_small
    backbone_model = vit_small(

                img_size = 518,
                patch_size = 14,
                init_values = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
                # num_register_tokens=4, # vits14_reg4
                num_register_tokens=0, # vits14
                interpolate_antialias = False,
                interpolate_offset = 0.1,
                )

    model_url = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vits14_pretrain.pth'
    state_dict = torch.load(model_url, map_location="cpu")
    backbone_model.load_state_dict(state_dict,strict=True)

    backbone_model.eval()
    backbone_model.cuda()
    # todo ----------------------------------#
    # todo load pretrained segmentation head
    import urllib

    # import mmcv
    # from mmcv.runner import load_checkpoint


    def load_config_from_url(url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()


    HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
    HEAD_TYPE = "ms" # in ("ms, "linear")


    # DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


    # head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    # head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"
    # head_config_url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_ms_config.py"
    # head_checkpoint_url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_ms_head.pth"


    head_config = '/home/lianghao/wangyushen/Projects/GaussTR/configs/dinov2/dinov2_vits14/dinov2_vits14_voc2012_ms_config.py'
    head_checkpoint = '/home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/dinov2_vits14_voc2012_ms_head.pth'



    # cfg_str = load_config_from_url(head_config_url)
    # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    from mmengine.config import Config
    cfg = Config.fromfile(head_config)

    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    model = create_segmenter(cfg, backbone_model=backbone_model)


    # load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    from mmengine.runner.checkpoint import _load_checkpoint,_load_checkpoint_to_model
    checkpoint = _load_checkpoint(head_checkpoint,map_location='cpu')
    checkpoint = _load_checkpoint_to_model(model,checkpoint,strict=False)

    model.cuda()
    model.eval()

    # todo ----------------------------------#
    # todo Load sample image
    import urllib
    from PIL import Image


    # def load_image_from_url(url: str) -> Image:
    #     with urllib.request.urlopen(url) as f:
    #         return Image.open(f).convert("RGB")


    # EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


    # image = load_image_from_url(EXAMPLE_IMAGE_URL)

    image_path = '/home/lianghao/wangyushen/Projects/GaussTR/data/example.jpg'
    image = Image.open(image_path).convert("RGB")


    # display(image)
    # todo ----------------------------------#
    # todo Semantic segmentation on sample image
    import numpy as np

    import dinov2.eval.segmentation.utils.colormaps as colormaps


    DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }


    def render_segmentation(segmentation_logits, dataset):
        colormap = DATASET_COLORMAPS[dataset]
        colormap_array = np.array(colormap, dtype=np.uint8)
        segmentation_values = colormap_array[segmentation_logits + 1]
        return Image.fromarray(segmentation_values)


    array = np.array(image)[:, :, ::-1] # BGR
    # segmentation_logits = inference_segmentor(model, array)[0]
    # from mmcv.transforms import MultiScaleFlipAug,Resize,RandomFlip,Normalize,ImageToTensor


    segmentation_logits = inference_model(model, array)[0]


    segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
    # display(segmented_image)




    # todo ----------------------------------#
    # todo Load pretrained segmentation model (Mask2Former)
    # import dinov2.eval.segmentation_m2f.models.segmentors

    # CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
    # CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

    # cfg_str = load_config_from_url(CONFIG_URL)
    # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    # model = init_segmentor(cfg)
    # load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    # model.cuda()
    # model.eval()
    # # todo ----------------------------------#
    # # todo Semantic segmentation on sample image
    # array = np.array(image)[:, :, ::-1] # BGR
    # segmentation_logits = inference_segmentor(model, array)[0]
    # segmented_image = render_segmentation(segmentation_logits, "ade20k")
    # display(segmented_image)