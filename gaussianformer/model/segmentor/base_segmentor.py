from mmdet3d.registry import MODELS
from mmengine.model import BaseModel, BaseModule, ModuleList

class CustomBaseSegmentor(BaseModel):

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.lifter = MODELS.build(lifter)
        self.encoder = MODELS.build(encoder)
        self.head = MODELS.build(head)


    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}

    def forward(
        self,
        imgs,
        metas,
        **kwargs
    ):
        pass