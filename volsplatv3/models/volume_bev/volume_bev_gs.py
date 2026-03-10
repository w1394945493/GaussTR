import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS





class MaxDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用不同尺度的池化来获取远近不同的特征
        self.max_p1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_p2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        # 通过最大池化，特征只会“扩张”，不会“变淡”
        feat_p1 = self.max_p1(x)
        feat_p2 = self.max_p2(x)
        
        # 将原始特征和池化后的特征融合
        # 这里的逻辑是：原值 > 3x3池化 > 5x5池化
        out = torch.max(x, feat_p1)
        out = torch.max(out, feat_p2)
        return out






@MODELS.register_module()
class VolumeGaussianBEV(BaseModule):
    def __init__(self,
                 encoder=None,
                 gs_decoder=None,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if encoder:
            self.encoder = MODELS.build(encoder)
        if gs_decoder:
            self.gs_decoder = MODELS.build(gs_decoder)
        
        # self.max_pool = MaxDiffusion()

        self.bev_h = self.encoder.bev_h
        self.bev_w = self.encoder.bev_w
        self.pc_range = self.encoder.pc_range
        
        # 预计算范围，提高 forward 效率
        self.pc_xrange = self.pc_range[3] - self.pc_range[0]
        self.pc_yrange = self.pc_range[4] - self.pc_range[1]

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, img_feats, candidate_gaussians, candidate_feats, img_metas=None, status="train"):

        if candidate_gaussians is not None and candidate_feats is not None: 
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            project_feats_hw = candidate_feats[0].new_zeros((bs, self.bev_h, self.bev_w, c))

            for i in range(bs):
                candidate_xyzs_i = candidate_gaussians[i][..., :3]
                candidate_hs_i = (self.bev_h * (candidate_xyzs_i[..., 1] - self.pc_range[1]) / self.pc_yrange - 0.5).int()
                candidate_ws_i = (self.bev_w * (candidate_xyzs_i[..., 0] - self.pc_range[0]) / self.pc_xrange - 0.5).int()
                
                candidate_feats_i = candidate_feats[i]
                
                candidate_coords_hw_i = torch.stack([candidate_hs_i, candidate_ws_i], dim=-1) # 将h和w合成二维坐标
                linear_inds_hw_i = (candidate_coords_hw_i[..., 0] * self.bev_w + candidate_coords_hw_i[..., 1]).to(dtype=torch.int64) # 将2D坐标展平为1维索引(H*W)
                project_feats_hw_i = project_feats_hw[i].view(-1, c)
                project_feats_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_hw_i = project_feats_hw_i.new_zeros((self.bev_h * self.bev_w, c), dtype=torch.float32)
                ones_hw_i = torch.ones_like(candidate_feats_i)
                count_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), ones_hw_i)
                count_hw_i = torch.where(count_hw_i == 0, torch.ones_like(count_hw_i), count_hw_i)
                project_feats_hw_i = (project_feats_hw_i / count_hw_i).view(self.bev_h, self.bev_w, c)
                project_feats_hw[i] = project_feats_hw_i
            
            project_feats_hw = rearrange(project_feats_hw, "b h w c -> b c h w")
            # ----------------------------------- #
            # todo 向邻域空间max_pool一下特征
            # project_feats_hw = self.max_pool(project_feats_hw)
            
            project_feats = [project_feats_hw]
        else:
            project_feats = [None]
        
        
        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas)
    
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            )

            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, outs, use_reentrant=False)
        else:
            outs = self.encoder(img_feats, project_feats, img_metas)
            gaussians = self.gs_decoder(outs)
        
        
        return gaussians