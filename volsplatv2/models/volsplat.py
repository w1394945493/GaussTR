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

import MinkowskiEngine as ME

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
                num_queries,
                
                transformer_decoder,
                regress_head,
                gauss_head,
                foreground_head,
                
                sparse_unet,
                sparse_gs,
                gaussian_adapter,
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
        
        
        self.neck = MODELS.build(neck)
        
        
        self.transformer_decoder = MODELS.build(transformer_decoder)
        self.query_embeds = nn.Embedding(
            num_queries, transformer_decoder.layer_cfg.self_attn_cfg.embed_dims) # self.query_embeds.weight.shape         
        self.reg_branches = nn.ModuleList([MODELS.build(regress_head) for _ in range(transformer_decoder.num_layers)])
        self.gauss_branches = nn.ModuleList([MODELS.build(gauss_head) for _ in range(transformer_decoder.num_layers)])
        self.foreground_head = MODELS.build(foreground_head)
        
        self.sparse_unet = MODELS.build(sparse_unet)
        self.gaussian_head = MODELS.build(sparse_gs)
        self.gaussian_adapter = MODELS.build(gaussian_adapter)
        self.voxel_resolution = voxel_resolution
        
        self.decoder = MODELS.build(decoder)

        
        
        

        print(cyan(f'successfully init Model!'))

    def _sparse_to_batched(self, features, coordinates, batch_size, return_mask=False):

        device = features.device
        _, c = features.shape

        batch_features_list = []
        batch_sizes = []
        max_voxels = 0

        for batch_idx in range(batch_size):
            mask = coordinates[:, 0] == batch_idx
            batch_feats = features[mask]  # [N_i, C]
            batch_features_list.append(batch_feats)
            batch_sizes.append(batch_feats.shape[0])
            max_voxels = max(max_voxels, batch_feats.shape[0])

        # Create padded tensor [b, 1, N_max, C]
        batched_features = torch.zeros(batch_size, 1, max_voxels, c, device=device)

        # Create valid data mask [b, 1, N_max]
        if return_mask:
            valid_mask = torch.zeros(batch_size, 1, max_voxels, dtype=torch.bool, device=device)

        for batch_idx, batch_feats in enumerate(batch_features_list):
            n_voxels = batch_feats.shape[0]
            batched_features[batch_idx, 0, :n_voxels, :] = batch_feats
            if return_mask:
                valid_mask[batch_idx, 0, :n_voxels] = True

        if return_mask:
            return batched_features, valid_mask
        return batched_features

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
        d_h, d_w = depth.shape[-2:]

        # todo ----------------------------------------#
        # todo 将深度图缩放和2D特征图尺寸一致
        img_feats = feats[0]  # (bv c h w)  
        f_h, f_w = img_feats.shape[-2:] 
        d_h, d_w = depth.shape[-2:]    
        if (d_w != f_w) or (d_h != f_h):
            depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
            d_h, d_w = depth.shape[-2:]
            
        
        img_aug_mat = data_samples['img_aug_mat']
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) 
        extrinsics = data_samples['cam2lidar'] #! surroundocc: 相机 -> lidar
        
        
        resize = torch.diag(torch.tensor([d_w/input_w, d_h/input_h],
                                        dtype=img_aug_mat.dtype,
                                        device = img_aug_mat.device))
        
        mat = torch.eye(4).to(img_aug_mat.device)            
        mat[:2,:2] = resize
        mat = repeat(mat,"i j -> () () i j")
        img_aug_mat = mat @ img_aug_mat  
        
        '''        
        # todo 这里检查一下depth, intrinsics, extrinsics, img_aug_mat 是否正确
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
        # todo 1. 体素化聚合像素特征，并筛选预测概率最大的前25600个特征点作为查询
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)  #! 外参：确定是cam2lidar还是cam2ego               
                img_feats,  # (bv c h w)
                depth=depth,
                # voxel_resolution=self.refine_voxel_resolution,
                voxel_resolution=0.2,
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                )  
              
        probs = self.foreground_head(sparse_input) # ((b n),dim)
        probs_params, valid_mask = self._sparse_to_batched(probs.F, probs.C, bs, return_mask=True)  # [b, 1, N_max, dim], [b, 1, N_max]
        batched_points = self._sparse_to_batched(aggregated_points, probs.C, bs) # (b 1 N_max,3)  
        batched_feats = self._sparse_to_batched(sparse_input.F, probs.C, bs)  # (b 1 N_max,128) 
        
        B, _, N, num_classes = probs_params.shape
        
        probs_softmax = F.softmax(probs_params, dim=-1)
        fg_probs, _ = probs_softmax[..., :num_classes-1].max(dim=-1)
        # fg_probs = 1 - probs_softmax[..., -1] # (b 1 n)
        k = min(25600, N)
        topk_probs, topk_indices = torch.topk(fg_probs, k=k, dim=-1)
        
        point_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, batched_points.shape[-1])
        selected_points = torch.gather(batched_points, dim=2, index=point_indices)
        
        feat_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, batched_feats.shape[-1])
        selected_feats = torch.gather(batched_feats, dim=2, index=feat_indices)
        
        selected_feats = rearrange(selected_feats,'b v n c -> b (v n) c')
        selected_points = rearrange(selected_points, 'b v n c -> b (v n) c')
        '''
        means3d = selected_points[0] # (num,c)
        import numpy as np
        # 保存为 numpy
        np.save("means3d_select_2e-1.npy", means3d.detach().cpu().numpy())
        '''
             
        
        
        decoder_inputs = self.pre_transformer(feats) 
        feats_flatten = flatten_multi_scale_feats(feats)[0]
        
        # todo ------------------------------------------------------------------#
        # todo 将筛选出来的25600个特征 + 固定数量的可学习特征，一同作为 目标查询query
        decoder_inputs.update(self.pre_decoder(feats_flatten)) # todo 准备查询query
        query, reference_points = self.forward_decoder(reg_branches=self.reg_branches,**decoder_inputs) # (num_layers,(bs,v),n_query,dim) (num_layers,(bs,v),n_query,3)  
        
        

        

      
        
        if mode == 'predict':
            query_gaussians = self.query_gaussian(-1,query[-1],reference_points[-1],depth,
                                                  intrinsics, extrinsics, img_aug_mat)            
            return self.decoder(query_gaussians,data,mode=mode)
        
        
        losses = {}
        num_layers = query.shape[0]
        for i in range(num_layers):
            query_gaussians = self.query_gaussian(i,query[i],reference_points[i],depth,
                                                  intrinsics, extrinsics, img_aug_mat)
            loss = self.decoder(query_gaussians,data,mode=mode)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses
                

        # gaussians = self.voxel_gaussian(bs,n,
        #                                 input_size=(input_h,input_w),
        #                                 feats=feats,data_samples=data_samples)
        
        # # todo 占用预测
        # return self.decoder(gaussians,data,mode=mode)

    
    
    
    
    
    def voxel_gaussian(self,
                       bs, n,
                       input_size,
                       feats,
                       data_samples):
        input_h,input_w = input_size
        
        img_feats = feats[0]  # (bv c h w)  

        
        depth = data_samples["depth"]  # (b v h w)
        # todo ----------------------------------------#
        # todo 将深度图缩放和2D特征图尺寸一致
        f_h, f_w = img_feats.shape[-2:] 
        d_h, d_w = depth.shape[-2:]    
        if (d_w != f_w) or (d_h != f_h):
            depth = F.interpolate(depth,size=(f_h,f_w),mode='bilinear',align_corners=False)
            
        # todo ----------------------------------------#
        # todo 将2D图像特征投影到3D体素空间
        img_aug_mat = data_samples['img_aug_mat']
        intrinsics = data_samples['cam2img'][...,:3,:3] # (b v 3 3) # todo 内参(相对于原图像)
        extrinsics = data_samples['cam2lidar'] # todo          
        
        
        # todo 这里用input_w, input_h没有问题，img_aug_mat是原图像到inputs的变换矩阵
        resize = torch.diag(torch.tensor([f_w/input_w, f_h/input_h],
                                        dtype=img_aug_mat.dtype,device = img_aug_mat.device))
        
        
        mat = torch.eye(4).to(img_aug_mat.device)            
        mat[:2,:2] = resize
        mat = repeat(mat,"i j -> () () i j")
        img_aug_mat = mat @ img_aug_mat        
        
        sparse_input, aggregated_points, counts = project_features_to_me(
                intrinsics, # (b v 3 3)
                extrinsics, # (b v 4 4)  #! 外参：确定是cam2lidar还是cam2ego               
                img_feats,  # (bv c h w)
                depth=depth,
                # voxel_resolution=self.refine_voxel_resolution,
                voxel_resolution=self.voxel_resolution,
                b=bs, v=n,
                normal=False,
                img_aug_mat=img_aug_mat,
                ) # sparse_input.C: (n,4) sparse_input.F: (n,128)         
        
        # todo -----------------------------------------------------#
        # todo 3D unet网络 进行细化
        sparse_out = self.sparse_unet(sparse_input)   # 3D Sparse UNet
        # todo 残差连接
        if torch.equal(sparse_out.C, sparse_input.C) and sparse_out.F.shape[1] == sparse_input.F.shape[1]: # todo sparse_out.C: (N,4) 4(batch_indices,x,y,z)
            # Create new feature tensor
            new_features = sparse_out.F + sparse_input.F # todo 见论文 3(C).1) Feature Refinement 的 公式(8)

            sparse_out_with_residual = ME.SparseTensor(
                features=new_features,
                coordinate_map_key=sparse_out.coordinate_map_key,
                coordinate_manager=sparse_out.coordinate_manager
            )
        else:
            # Handle coordinate mismatch
            print("Warning: Input and output coordinates inconsistent, skipping residual connection")
            sparse_out_with_residual = sparse_out        
        
        # todo ------------------------------------------------------------------------#
        # todo 高斯参数预测
        gaussians = self.gaussian_head(sparse_out_with_residual)
        del sparse_out_with_residual,sparse_out,sparse_input,new_features
        

        # todo ----------------------#
        # todo  这里进行了逐batch处理
        gaussian_params, valid_mask = self._sparse_to_batched(gaussians.F, gaussians.C, bs, return_mask=True)  # [b, 1, N_max, 38], [b, 1, N_max]
        batched_points = self._sparse_to_batched(aggregated_points, gaussians.C, bs)  # [b, 1, N_max, 3]        

        opacity_raw = gaussian_params[..., :1]  # [b, 1, N_max, 1]
        opacity_raw = torch.where(
            valid_mask.unsqueeze(-1),  # [b, 1, N_max, 1]
            opacity_raw,
            torch.full_like(opacity_raw, -20.0)  # sigmoid(-20) ≈ 2e-9，
        ) # todo 这里为了能保证多bs处理，将无效特征的透明度用-20填充了
        opacities = opacity_raw.sigmoid().unsqueeze(-1)  #[b, 1, N_max, 1, 1]
        raw_gaussians = gaussian_params[..., 1:]    #[b, 1, N_max, 37]
        raw_gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c",srf=1,)
        
        # todo 预测高斯后处理
        gaussians = self.gaussian_adapter.forward(
            opacities = opacities,   # (b 1 n 1 1)
            raw_gaussians = (raw_gaussians,"b v r srf c -> b v r srf () c"), # (b 1 n 1 1 c)
            points = batched_points, # (b 1 n 3)
            voxel_resolution = self.voxel_resolution, #! 体素网格尺寸 
        )        
        
        
        return gaussians
    
    
    def query_gaussian(self, i,
                       x,ref_pts,depth,
                       intrinsics, extrinsics, img_aug_mat):
        

        bs, n, d_w, d_h = depth.shape

        x = x.reshape(bs, n, *x.shape[1:])
        deltas = self.reg_branches[i](x)
        ref_pts = (deltas[..., :2] + inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid()            
        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],ref_pts.unsqueeze(2) * 2 - 1) 
        sample_depth = sample_depth[:, :, 0, 0, :, None] # (b v 1 1 n) -> (b v n 1)
        
        # todo d_w 和 d_h
        points = torch.cat([ref_pts * torch.tensor([d_w, d_h]).to(x), sample_depth * (1 + deltas[..., 2:3])], -1)
        means = cam2world(points, intrinsics, extrinsics, img_aug_mat) # (b v n 3)
        
        raw_gaussians = self.gauss_branches[i](x) # (b v n 1+3+4+3+18=29)
        sh_dim = 3 * self.gaussian_adapter.d_sh if self.gaussian_adapter.d_sh is not None else 3
        fixed_len = 1 + 3 + 4 + sh_dim
        fixed_part, semantics = raw_gaussians.split([fixed_len, raw_gaussians.shape[-1] - fixed_len], dim=-1)
        opacity_raw, scales, rotations, sh = fixed_part.split((1, 3, 4, sh_dim), dim=-1)
        
        opacities = opacity_raw.sigmoid().unsqueeze(-1) # (b v n 1 1)
        if self.gaussian_adapter.d_sh:
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)    # [b, v, n, (xyz d_sh)] -> [b, v, n, xyz, d_sh]
            sh = sh.broadcast_to((*opacities.shape, 3, self.gaussian_adapter.d_sh)) * self.gaussian_adapter.sh_mask
        else:
            sh = torch.sigmoid(sh)            
        
        scales = self.gaussian_adapter.gaussian_scale_min + (self.gaussian_adapter.gaussian_scale_max - self.gaussian_adapter.gaussian_scale_min) * torch.sigmoid(scales) # (b v n 3)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8) # (b v n 4)
        semantics = F.softplus(semantics) # (b v n 18)
        covariances = build_covariance(scales, rotations) # (b v n 3 3)
        c2w_rotations = extrinsics[..., :3, :3] # (b v 3 3)         
        c2w_rotations = rearrange(c2w_rotations,"b v i j -> b v () i j") 
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2) # (b v n 3 3)
        
        gaussians = Gaussians(rearrange(means,"b v n xyz -> b (v n) xyz"), 
            rearrange(scales,"b v n xyz -> b (v n) xyz"), 
            rearrange(rotations,"b v n d -> b (v n) d"),                             
            rearrange(covariances,"b v n i j -> b (v n) i j",), 
            rearrange(sh,"b v n c d_sh -> b (v n) c d_sh",) \
                if self.gaussian_adapter.d_sh is not None else rearrange(sh,"b v n rgb -> b (v n) rgb",),
            rearrange(opacities,   "b v n srf spp -> b (v n srf spp)"), 
            rearrange(semantics,"b v n dim -> b (v n) dim")       
        )      
        return gaussians
    
    
    
    
    
    def pre_transformer(self, mlvl_feats):
        batch_size = mlvl_feats[0].size(0)

        mlvl_masks = []
        for feat in mlvl_feats:
            mlvl_masks.append(None)

        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        
        return decoder_inputs_dict

    def pre_decoder(self, memory):
        bs, _, c = memory.shape
        
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1) # todo 可学习的特征嵌入
        reference_points = torch.rand((bs, query.size(1), 2)).to(query) # todo 随机的0-1之间的参考点
        decoder_inputs_dict = dict(
            query=query, memory=memory, reference_points=reference_points)
        return decoder_inputs_dict
    
    def forward_decoder(self, query, memory, memory_mask, reference_points,
                        spatial_shapes, level_start_index, valid_ratios,
                        **kwargs):
        inter_states, references = self.transformer_decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        return inter_states, references