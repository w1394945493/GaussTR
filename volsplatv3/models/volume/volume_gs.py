import torch
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS







@MODELS.register_module()
class VolumeGaussian(BaseModule):
    def __init__(self,
                 encoder=None,
                 gs_decoder=None,
                 use_checkpoint=False,
                 **kwargs,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        if encoder:
            self.encoder = MODELS.build(encoder)
        if gs_decoder:
            self.gs_decoder = MODELS.build(gs_decoder)

        self.tpv_h = self.encoder.tpv_h
        self.tpv_w = self.encoder.tpv_w
        self.tpv_z = self.encoder.tpv_z
        self.pc_range = self.encoder.pc_range
        self.pc_xrange = self.pc_range[3] - self.pc_range[0]
        self.pc_yrange = self.pc_range[4] - self.pc_range[1]
        self.pc_zrange = self.pc_range[5] - self.pc_range[2]
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    # todo：体素高斯预测：三维体素投影 + TPV表征 + 高斯点解码
    def forward(self, img_feats, candidate_gaussians, candidate_feats, img_metas=None, status="train"):
        """Forward training function.
        """
        if candidate_gaussians is not None and candidate_feats is not None: # candidate_gaussians: 高斯点(14维) candidate_feats：高斯特征(128维)
            bs = len(candidate_feats)
            _, c = candidate_feats[0].shape
            # todo-----------------------------#
            # todo 1. TPV(三维投影体素)映射：将三维点云投影到三个平面: tpv_h,tpv_w,tpv_z: 192, 192, 16
            project_feats_hw = candidate_feats[0].new_zeros((bs, self.tpv_h, self.tpv_w, c)) # HW平面，沿Z方向投影 张量数据类型与设备和candidate_feats一致
            project_feats_zh = candidate_feats[0].new_zeros((bs, self.tpv_z, self.tpv_h, c)) # ZH平面，沿Y方向投影
            project_feats_wz = candidate_feats[0].new_zeros((bs, self.tpv_w, self.tpv_z, c)) # WZ平面，沿X方向投影

            for i in range(bs): # 经筛选后，每个bs的像素高斯数量不一致
                candidate_xyzs_i = candidate_gaussians[i][..., :3]
                # todo: 将每个高斯点的xyz坐标归一化到TPV网格索引
                # pc_range: 点云的有效空间范围；(tpv_h,tpv_w,tpv_z)：TPV平面网格的尺寸(格子数)
                # 把y坐标归一化到0-1之间((y-pc_yrange)/pc_yrange)，再将归一化值映射到TPV高度格子索引[0,tpv_h-1], -0.5:对齐TPV格子中心
                candidate_hs_i = (self.tpv_h * (candidate_xyzs_i[..., 1] - self.pc_range[1]) / self.pc_yrange - 0.5).int()
                # 同理，x坐标归一化、对齐过程
                candidate_ws_i = (self.tpv_w * (candidate_xyzs_i[..., 0] - self.pc_range[0]) / self.pc_xrange - 0.5).int()
                # 同理，z坐标归一化、对齐过程
                candidate_zs_i = (self.tpv_z * (candidate_xyzs_i[..., 2] - self.pc_range[2]) / self.pc_zrange - 0.5).int()
                # n, c
                #candidate_feats_i = candidate_feats[[i, valid_mask]]
                candidate_feats_i = candidate_feats[i]
                # todo：将不规则高斯点特征映射到规则平面网格上，便于卷积操作处理：
                # hw: n, 2
                # 将高斯点特征投影到HW平面上，并对重叠格子做平均
                candidate_coords_hw_i = torch.stack([candidate_hs_i, candidate_ws_i], dim=-1) # 将h和w合成二维坐标
                linear_inds_hw_i = (candidate_coords_hw_i[..., 0] * self.tpv_w + candidate_coords_hw_i[..., 1]).to(dtype=torch.int64) # 2D坐标展平为1维索引(H*W)
                project_feats_hw_i = project_feats_hw[i].view(-1, c)
                project_feats_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), candidate_feats_i) # 将每个高斯点特征加到对应的格子，多个点落在同一个格子，则会叠加
                count_hw_i = project_feats_hw_i.new_zeros((self.tpv_h * self.tpv_w, c), dtype=torch.float32)
                ones_hw_i = torch.ones_like(candidate_feats_i)
                count_hw_i.scatter_add_(0, linear_inds_hw_i.unsqueeze(-1).expand(-1, c), ones_hw_i)
                count_hw_i = torch.where(count_hw_i == 0, torch.ones_like(count_hw_i), count_hw_i) # 统计每个格子累加了多少个高斯点(避免除0)
                project_feats_hw_i = (project_feats_hw_i / count_hw_i).view(self.tpv_h, self.tpv_w, c)
                project_feats_hw[i] = project_feats_hw_i # 对累加特征取平均，得到最终的HW平面特征图

                # zh: n, 2
                candidate_coords_zh_i = torch.stack([candidate_zs_i, candidate_hs_i], dim=-1)
                linear_inds_zh_i = (candidate_coords_zh_i[..., 0] * self.tpv_h + candidate_coords_zh_i[..., 1]).to(dtype=torch.int64)
                project_feats_zh_i = project_feats_zh[i].view(-1, c)
                project_feats_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_zh_i = project_feats_zh_i.new_zeros((self.tpv_z * self.tpv_h, c), dtype=torch.float32)
                ones_zh_i = torch.ones_like(candidate_feats_i)
                count_zh_i.scatter_add_(0, linear_inds_zh_i.unsqueeze(-1).expand(-1, c), ones_zh_i)
                count_zh_i = torch.where(count_zh_i == 0, torch.ones_like(count_zh_i), count_zh_i)
                project_feats_zh_i = (project_feats_zh_i / count_zh_i).view(self.tpv_z, self.tpv_h, c)
                project_feats_zh[i] = project_feats_zh_i

                # wz: n, 2
                candidate_coords_wz_i = torch.stack([candidate_ws_i, candidate_zs_i], dim=-1)
                linear_inds_wz_i = (candidate_coords_wz_i[..., 0] * self.tpv_z + candidate_coords_wz_i[..., 1]).to(dtype=torch.int64)
                project_feats_wz_i = project_feats_wz[i].view(-1, c)
                project_feats_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), candidate_feats_i)
                count_wz_i = project_feats_wz_i.new_zeros((self.tpv_w * self.tpv_z, c), dtype=torch.float32)
                ones_wz_i = torch.ones_like(candidate_feats_i)
                count_wz_i.scatter_add_(0, linear_inds_wz_i.unsqueeze(-1).expand(-1, c), ones_wz_i)
                count_wz_i = torch.where(count_wz_i == 0, torch.ones_like(count_wz_i), count_wz_i)
                project_feats_wz_i = (project_feats_wz_i / count_wz_i).view(self.tpv_w, self.tpv_z, c)
                project_feats_wz[i] = project_feats_wz_i

            project_feats_hw = rearrange(project_feats_hw, "b h w c -> b c h w")
            project_feats_zh = rearrange(project_feats_zh, "b h w c -> b c h w")
            project_feats_wz = rearrange(project_feats_wz, "b h w c -> b c h w")
            project_feats = [project_feats_hw, project_feats_zh, project_feats_wz]
        else:
            project_feats = [None, None, None]

        if self.use_checkpoint and status != "test":
            input_vars_enc = (img_feats, project_feats, img_metas) # img_feats: 下采样1/4的二维图像特征图 project_feats: TPV特征

            # todo -----------------------------------------------------------#
            # todo encoder: 把多视角图像特征整合成统一的三维空间特征表示
            # 把3D场景分成三个平面(HW,ZH,WZ),在每个平面上构建token query; 利用图像特征等来更新token，最终形成体素级别的空间表示
            outs = torch.utils.checkpoint.checkpoint(
                self.encoder, *input_vars_enc, use_reentrant=False
            ) # 用梯度检查直至调用编码器，来节省显存 outs 就是self.encoder(img_feats,project_feats,img_meats)的输出，通过checkpoint机制执行，显存开销更低

            # todo 从三平面特征(TPV)生成3D高斯体: 将每个体素位置的特征解码为可渲染的3D高斯体，用于重建三维场景
            gaussians = torch.utils.checkpoint.checkpoint(self.gs_decoder, outs, use_reentrant=False)
        else:
            outs = self.encoder(img_feats, project_feats, img_metas)
            gaussians = self.gs_decoder(outs)
        bs = gaussians.shape[0]
        n_feature = gaussians.shape[-1] # 14
        gaussians = gaussians.reshape(bs, -1, n_feature) # 展平
        return gaussians
        