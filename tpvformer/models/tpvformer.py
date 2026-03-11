

import torch
from einops import rearrange,repeat

from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

from .utils.me_fea import project_features

torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@MODELS.register_module()
class TPVFormer(BaseModel):
    def __init__(self,
                 
                backbone,
                neck,    
                volume,
                decoder,
                
                use_checkpoint,
                voxel_resolution,
                pc_range,
                **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

        self.volume = MODELS.build(volume)
        self.decoder = MODELS.build(decoder)
        self.use_checkpoint = use_checkpoint
        self.voxel_resolution = voxel_resolution
        self.pc_range = pc_range
        print(cyan(f'successfully init Model!'))

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        
        img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def forward(self, mode='loss',**data):
        
        inputs = data['img']
        bs,n,_,h,w = inputs.shape
        data_samples = data
    
        img_feats = self.extract_img_feat(img=inputs)
        
        
        depth = data_samples["depth"] # (b v 112 200)
        
        img_aug_mat = data_samples['img_aug_mat'] #! ori_img -> inputs
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) #! cam -> ori_img
        extrinsics = data_samples['cam2lidar']

        bs, n = intrinsics.shape[:2]

        pixel_points,pixel_feats = project_features(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)    
                rearrange(img_feats[0], "b v c h w -> (b v) c h w"),  # (bv c h w)
                depth=depth,           
        )

        x_start, y_start, z_start, x_end, y_end, z_end = self.pc_range # todo [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        candidate_pos_mask, candidate_feat_mask = [], []

        positions = pixel_points  # 使用像素特征
        feats = pixel_feats
    
        for b in range(bs):
            mask_pixel_i = (positions[b, :, 0] >= x_start) & (positions[b, :, 0] <= x_end) & \
                        (positions[b, :, 1] >= y_start) & (positions[b, :, 1] <= y_end) & \
                        (positions[b, :, 2] >= z_start) & (positions[b, :, 2] <= z_end)            
            candidate_pos_mask_i = positions[b][mask_pixel_i]
            candidate_feat_mask_i = feats[b][mask_pixel_i]
            candidate_pos_mask.append(candidate_pos_mask_i)
            candidate_feat_mask.append(candidate_feat_mask_i)  
        
        img_meats = {
            'img_shape': list(img_feats[0].shape[-2:]),
            'lidar2img': data_samples['projection_mat'], # (1 6 4 4)
        }

        # 体素语义预测
        outputs = self.volume(
                [img_feats[0]],
                candidate_pos_mask,
                candidate_feat_mask,
                img_meats,
                ) 
        
        return self.decoder(outputs, data, mode=mode)