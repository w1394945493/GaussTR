import torch
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS

@MODELS.register_module()
class VolumeBEVGaussian(BaseModule):
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
                
                # 特征
                candidate_feats_i = candidate_feats[i]
                
                # 将高斯点特征投影到HW平面上，并对重叠格子做平均
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