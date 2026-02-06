from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS

from einops import rearrange,repeat
import torch.nn.functional as F
from math import isqrt

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# import MinkowskiEngine as ME

from mmdet.models import inverse_sigmoid

from .encoder.common.me_fea import project_features_to_me
from .utils.types import Gaussians

from .utils import flatten_multi_scale_feats, flatten_bsn_forward,cam2world
from .encoder.common.gaussians import build_covariance
from ..geometry.projection import sample_image_grid,get_world_rays

torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

@MODELS.register_module()
class VolSplat(BaseModel):

    def __init__(self,
                # backbone,
                neck,
                
                model_url,
                top_k,
                foreground_head,
                lifter,
                encoder,
                
                # sparse_unet,
                # sparse_gs,
                # gaussian_adapter,
                decoder,

                use_checkpoint,
                # refine_voxel_resolution,
                voxel_resolution,
                
                 **kwargs):
        super().__init__(**kwargs)

        # self.backbone = MODELS.build(backbone)
        self.use_checkpoint = use_checkpoint
        
        
        from dinov2.models.vision_transformer import vit_base
        self.backbone = vit_base(

            img_size = 518,
            patch_size = 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            num_register_tokens=4,
            interpolate_antialias = False,
            interpolate_offset = 0.1,
            )
        state_dict = torch.load(model_url, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        print(cyan(f"load checkpoint from{model_url}."))
        self.backbone.requires_grad_(False)
        self.backbone.is_init = True
        self.patch_size = self.backbone.patch_size
        
        self.top_k = top_k
        self.neck = MODELS.build(neck)
        
        self.foreground_head = MODELS.build(foreground_head)
        self.lifter = MODELS.build(lifter)
        self.encoder = MODELS.build(encoder)
        
        # self.sparse_unet = MODELS.build(sparse_unet)
        # self.gaussian_head = MODELS.build(sparse_gs)
        # self.gaussian_adapter = MODELS.build(gaussian_adapter)
        
        
        self.voxel_resolution = voxel_resolution
        
        self.decoder = MODELS.build(decoder)
        print(cyan(f'successfully init Model!'))

    def _sparse_to_batched(self, features, coordinates, batch_size, return_mask=False):

        device = features.device
        num_voxels, c = features.shape

        # todo -----------------------------------#
        batch_indices = coordinates[:, 0].long()
        v_counts = torch.bincount(batch_indices, minlength=batch_size) # [batch_size]
        max_voxels = v_counts.max().item()
        
        order = torch.arange(num_voxels, device=device)
        # 按照 batch_indices 排序后的偏移量
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        batch_offsets[1:] = torch.cumsum(v_counts, dim=0)
        # 计算每个点在自己 batch 内的相对位置索引 [0, 1, 2, ..., n_i-1]
        local_idx = order - batch_offsets[batch_indices]
        
        # 初始化稠密张量 [b, 1, N_max, C]
        batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)
        batched_features[batch_indices, 0, local_idx] = features
        if return_mask:
            valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)
            valid_mask[batch_indices, 0, local_idx] = True
            return batched_features, valid_mask
        return batched_features
        
        
        # batch_features_list = []
        # batch_sizes = []
        # max_voxels = 0

        # for batch_idx in range(batch_size):
        #     mask = coordinates[:, 0] == batch_idx
        #     batch_feats = features[mask]  # [N_i, C]
        #     batch_features_list.append(batch_feats)
        #     batch_sizes.append(batch_feats.shape[0])
        #     max_voxels = max(max_voxels, batch_feats.shape[0])
        # # Create padded tensor [b, 1, N_max, C]
        # batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)

        # # Create valid data mask [b, 1, N_max]
        # if return_mask:
        #     valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)

        # for batch_idx, batch_feats in enumerate(batch_features_list):
        #     n_voxels = batch_feats.shape[0]
        #     batched_features[batch_idx, 0, :n_voxels, :] = batch_feats
        #     if return_mask:
        #         valid_mask[batch_idx, 0, :n_voxels] = True

        # if return_mask:
        #     return batched_features, valid_mask
        # return batched_features

    def extract_img_feat(self, img, status="train"):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_checkpoint and status != "test":
            img_feats = torch.utils.checkpoint.checkpoint(
                            self.backbone, img, use_reentrant=False)
        else:
            img_feats = self.backbone(img)
        img_feats = self.neck(img_feats) # BV, C, H, W
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped
    
    def forward(self, mode='loss',**data):
        
        inputs = data['img']
        data_samples = data

        bs, n, _, input_h, input_w = inputs.shape # (b,v,3,H,W)
        
        
        
        # todo backbone+FPN特征提取
        # multi_img_feats = self.extract_img_feat(img=inputs)
        # img_feats = rearrange(multi_img_feats[0], "b v c h w -> (b v) c h w")
        
        # todo vit + FPN特征提取
        concat = rearrange(inputs,"b v c h w -> (b v) c h w")
        resize_h, resize_w = input_h // self.patch_size * self.patch_size, input_w // self.patch_size * self.patch_size
        concat = F.interpolate(concat,(resize_h,resize_w),mode='bilinear',align_corners=True)
        
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone.forward_features(
                concat)['x_norm_patchtokens']
            x = x.mT.reshape(bs * n, -1,
                                concat.shape[-2] // self.patch_size,
                                concat.shape[-1] // self.patch_size)
        feats = self.neck(x)   
        
        # todo transformer 解码器 -> 解码得到查询特征和特征在2D图像的位置
        depth = data_samples["depth"]  # (b v h w)
        # d_h, d_w = depth.shape[-2:]

        # todo ----------------------------------------#
        # todo 将深度图缩放和2D特征图尺寸一致
        img_feats = feats[0]  # todo (6 128 128 228)
        # f_h, f_w = img_feats.shape[-2:] 
        # d_h, d_w = depth.shape[-2:]    
        # if (d_w != f_w) or (d_h != f_h):
        #     depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False) # todo (1 6 128 228)
        #     d_h, d_w = depth.shape[-2:] # todo 128 228
            
        
        img_aug_mat = data_samples['img_aug_mat'] #! ori_img -> inputs
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) #! cam -> ori_img
        extrinsics = data_samples['cam2lidar'] #! surroundocc: cam -> lidar
        
        # resize = torch.diag(torch.tensor([d_w/input_w, d_h/input_h], # todo 228/800 128/448
        #                                 dtype=img_aug_mat.dtype,
        #                                 device = img_aug_mat.device)) # todo [[0.2850,0],[0,0.2857]]
        
        # mat = torch.eye(4).to(img_aug_mat.device)            
        # mat[:2,:2] = resize
        # mat = repeat(mat,"i j -> () () i j")
        # img_aug_mat = mat @ img_aug_mat  #! ori_img -> img_feats
        
        # todo 检查一下depth, intrinsics, extrinsics, img_aug_mat 是否正确
        '''        
        
        device = depth.device
        coordinates = sample_image_grid((d_h, d_w), device,normal=False)[0]
        coordinates = repeat(coordinates, "h w c -> 1 v (h w) c", v=depth.shape[1])
        # 齐次化 (b, v, n, 3)
        coordinates = torch.cat([coordinates,  torch.ones_like(coordinates[..., :1])], dim=-1) # (b, v, n, 3)
        post_rots = img_aug_mat[..., :3, :3] # (b,v,3,3)
        post_trans = img_aug_mat[..., :3, 3] # (b,v,3)    
        # 逆平移
        coordinates = coordinates - post_trans.unsqueeze(-2)
        # 逆旋转 (b, v, n, 3)
        coordinates = (torch.inverse(post_rots).unsqueeze(2) @ coordinates.unsqueeze(-1)).squeeze(-1)
        # 去掉齐次位并对齐维度 (b, v, n, 1, 1, 2)
        coordinates = rearrange(coordinates[...,:-1], "b v n xy -> b v n () () xy")
        intrinsics_ = rearrange(intrinsics, "b v i j -> b v () () () i j")
        extrinsics_ = rearrange(extrinsics, "b v i j -> b v () () () i j")
        origins, directions = get_world_rays(coordinates, extrinsics_, intrinsics_)
        depths = rearrange(depth, "b v h w -> b v (h w) () ()")
        world_coords = origins + directions * depths[..., None] # todo 计算得到每个像素的3D坐标
        world_coords = world_coords.squeeze(3).squeeze(3)         
        points = rearrange(world_coords, "b v n c -> b (v n) c")
        means3d = points[0] # (num,c)
        import numpy as np
        # 保存为 numpy
        np.save("means3d.npy", means3d.detach().cpu().numpy())
        '''        
        
        # todo ----------------------------------------------------------#
        # todo 1. 体素化聚合像素特征，并筛选预测概率最大的前top_k个实例作为查询先验
        anchor, instance_feature = self.lifter(bs)
        if self.top_k > 0:
            topk_anchor, topk_instance_feature = self.select_topk_instance(intrinsics,extrinsics,
                                                                        img_feats,depth,
                                                                        img_aug_mat,
                                                                        top_k=self.top_k)
            # todo 前top_k 个先验与 n个可学习token cat 共同作为 解码的目标查询
            anchor = torch.cat([topk_anchor,anchor],dim=1)
            instance_feature = torch.cat([topk_instance_feature,instance_feature],dim=1)  
        
        # anchor = topk_anchor
        # instance_feature = topk_instance_feature
        
        # todo ----------------------------------------------------------#
        # todo 初始的anchor和instance_feature也解码得到高斯点属性        
        '''
        means = anchor[1][...,:3] # (num,c)
        import numpy as np
        # 保存为 numpy
        np.save(f"means3d_total_6_1.npy", means.detach().cpu().numpy())
        '''      
        
        # todo ----------------------------------------------------------#
        # todo 2. 可变形多尺度特征聚合, 预测高斯点属性
        # todo 参数计算(可以放到dataset中处理)
        # lidar2cam = torch.inverse(extrinsics) # todo (1 6 4 4)
        # cam2img = data_samples['cam2img'] # todo (1 6 4 4)
        # projection_mat = img_aug_mat @ cam2img @ lidar2cam # todo (1 6 4 4)
        # featmap_wh = img_aug_mat.new_tensor([f_w,f_h])
        # featmap_wh = repeat(featmap_wh,"wh -> bs n wh",bs=bs,n=n) # todo (1 6 2)        
        projection_mat = data_samples['projection_mat']
        featmap_wh = data_samples['featmap_wh']
        predictions = self.encoder(anchor, instance_feature, feats, projection_mat, featmap_wh)

        if mode == 'predict':
            
            
            
            
            
            return self.decoder(predictions[-1],data,mode=mode)
        
        losses = {}
        for i in range(len(predictions)):
            loss = self.decoder(predictions[i],data,mode=mode)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        
        return losses
                

    
    def select_topk_instance(self,intrinsics,extrinsics,img_feats,depth,img_aug_mat,top_k=25600):
        bs,n = intrinsics.shape[:2]
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)    
                img_feats,  # (bv c h w)
                depth=depth,
                
                voxel_resolution=self.voxel_resolution, 
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                )          
        
        
        anchors_pred = self.foreground_head(sparse_input) # todo ((b n),18)
        
        # batched_anchors, valid_mask = self._sparse_to_batched(anchors_pred.F, anchors_pred.C, bs, return_mask=True)  # [b, 1, N_max, dim], [b, 1, N_max]
        
        # batched_points = self._sparse_to_batched(aggregated_points, anchors_pred.C, bs) # (b 1 N_max,3)  
        # batched_feats = self._sparse_to_batched(sparse_input.F, anchors_pred.C, bs)  # (b 1 N_max,128) 
        batched_anchors, valid_mask = self._sparse_to_batched(anchors_pred.features, anchors_pred.indices, bs, return_mask=True)  # [b, 1, N_max, dim], [b, 1, N_max]
        
        batched_points = self._sparse_to_batched(aggregated_points, anchors_pred.indices, bs) # (b 1 N_max,3)  
        batched_feats = self._sparse_to_batched(sparse_input.features, anchors_pred.indices, bs)  # (b 1 N_max,128) 
        
        
        
        
        batched_probs = batched_anchors[...,14:]
        B, _, N, num_classes = batched_probs.shape
        
        probs_softmax = F.softmax(batched_probs, dim=-1)
        fg_probs, _ = probs_softmax[..., :num_classes-1].max(dim=-1)
        # fg_probs = 1 - probs_softmax[..., -1] # (b 1 n)
        top_k = min(top_k, N)
        topk_probs, topk_indices = torch.topk(fg_probs, k=top_k, dim=-1)
        
        anchor_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, batched_anchors.shape[-1])
        selected_anchors = torch.gather(batched_anchors, dim=2, index=anchor_indices) 
        
        point_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, batched_points.shape[-1])
        selected_points = torch.gather(batched_points, dim=2, index=point_indices)
        
        feat_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, batched_feats.shape[-1])
        selected_feats = torch.gather(batched_feats, dim=2, index=feat_indices)
        
        selected_anchors = rearrange(selected_anchors, 'b v n c -> b (v n) c') # todo (1 25600 32)
        selected_points = rearrange(selected_points, 'b v n c -> b (v n) c') # todo (1 25600 3)
        selected_feats = rearrange(selected_feats,'b v n c -> b (v n) c') # todo (1 25600 128)
        
 
        
        offset_xyz = selected_anchors[...,:3]
        offset_xyz = offset_xyz.sigmoid()
        offset_world = (offset_xyz - 0.5) *self.voxel_resolution*3
        means = selected_points + offset_world
        anchor = torch.cat([means, selected_anchors[...,3:]],dim=2)
        
        instance_feature = selected_feats                

        return anchor, instance_feature
    


    # def voxel_gaussian(self,
    #                    bs, n,
    #                    input_size,
    #                    feats,
    #                    data_samples):
    #     input_h,input_w = input_size
        
    #     img_feats = feats[0]  # (bv c h w)  

        
    #     depth = data_samples["depth"]  # (b v h w)
    #     # todo ----------------------------------------#
    #     # todo 将深度图缩放和2D特征图尺寸一致
    #     f_h, f_w = img_feats.shape[-2:] 
    #     d_h, d_w = depth.shape[-2:]    
    #     if (d_w != f_w) or (d_h != f_h):
    #         depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
            
    #     # todo ----------------------------------------#
    #     # todo 将2D图像特征投影到3D体素空间
    #     img_aug_mat = data_samples['img_aug_mat']
    #     intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) # todo 内参(相对于原图像)
    #     extrinsics = data_samples['cam2lidar'] # todo          
        
        
    #     # todo 这里用input_w, input_h没有问题，img_aug_mat是原图像到inputs的变换矩阵
    #     resize = torch.diag(torch.tensor([f_w/input_w, f_h/input_h],
    #                                     dtype=img_aug_mat.dtype,device = img_aug_mat.device))
        
        
    #     mat = torch.eye(4).to(img_aug_mat.device)            
    #     mat[:2,:2] = resize
    #     mat = repeat(mat,"i j -> () () i j")
    #     img_aug_mat = mat @ img_aug_mat        
        
    #     sparse_input, aggregated_points, counts = project_features_to_me(
    #             intrinsics, # (b v 3 3)
    #             extrinsics, # (b v 4 4)  #! 外参：确定是cam2lidar还是cam2ego               
    #             img_feats,  # (bv c h w)
    #             depth=depth,
    #             # voxel_resolution=self.refine_voxel_resolution,
    #             voxel_resolution=self.voxel_resolution,
    #             b=bs, v=n,
    #             normal=False,
    #             img_aug_mat=img_aug_mat,
    #             ) # sparse_input.C: (n,4) sparse_input.F: (n,128)         
        
    #     # todo -----------------------------------------------------#
    #     # todo 3D unet网络 进行细化
    #     sparse_out = self.sparse_unet(sparse_input)   # 3D Sparse UNet
    #     # todo 残差连接
    #     if torch.equal(sparse_out.C, sparse_input.C) and sparse_out.F.shape[1] == sparse_input.F.shape[1]: # todo sparse_out.C: (N,4) 4(batch_indices,x,y,z)
    #         # Create new feature tensor
    #         new_features = sparse_out.F + sparse_input.F # todo 见论文 3(C).1) Feature Refinement 的 公式(8)

    #         sparse_out_with_residual = ME.SparseTensor(
    #             features=new_features,
    #             coordinate_map_key=sparse_out.coordinate_map_key,
    #             coordinate_manager=sparse_out.coordinate_manager
    #         )
    #     else:
    #         # Handle coordinate mismatch
    #         print("Warning: Input and output coordinates inconsistent, skipping residual connection")
    #         sparse_out_with_residual = sparse_out        
        
    #     # todo ------------------------------------------------------------------------#
    #     # todo 高斯参数预测
    #     gaussians = self.gaussian_head(sparse_out_with_residual)
    #     del sparse_out_with_residual,sparse_out,sparse_input,new_features
        

    #     # todo ----------------------#
    #     # todo  这里进行了逐batch处理
    #     gaussian_params, valid_mask = self._sparse_to_batched(gaussians.F, gaussians.C, bs, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
    #     batched_points = self._sparse_to_batched(aggregated_points, gaussians.C, bs)  # [b, 1, N_max, 3]        

    #     opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
    #     opacity_raw = torch.where(
    #         valid_mask.unsqueeze(-1),  # [b, 1, N_max, 1]
    #         opacity_raw,
    #         torch.full_like(opacity_raw, -20.0)  # sigmoid(-20) ≈ 2e-9，
    #     ) # todo 这里为了能保证多bs处理，将无效特征的透明度用-20填充了
    #     opacities = opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
    #     raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
    #     raw_gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c",srf=1,)
        
    #     # todo 预测高斯后处理
    #     gaussians = self.gaussian_adapter.forward(
    #         opacities = opacities,   # (b 1 n 1 1)
    #         raw_gaussians = (raw_gaussians,"b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
    #         points = batched_points, # (b 1 n 3)
    #         voxel_resolution = self.voxel_resolution, #! 体素网格尺寸 
    #     )        
        
        
    #     return gaussians
    
    
    

