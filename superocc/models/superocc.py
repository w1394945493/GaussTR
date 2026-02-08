import queue
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from .utils.aug import GpuPhotoMetricDistortion


@MODELS.register_module()
class SuperOCC(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 seq_mode=False,
                 pretrained=None,
                 **kwargs):
        super(SuperOCC, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)        
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()

        self.prev_scene_token = None
        self.seq_mode = seq_mode
        self.memory = {}
        self.queue = queue.Queue()

    def forward(self, mode='loss',**data):
        
        
        return