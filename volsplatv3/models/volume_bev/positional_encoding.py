import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmengine.registry import MODELS


@MODELS.register_module()
class BEVFormerPositionalEncoding(BaseModule):

    def __init__(self,
                 num_feats,
                 h,
                 w,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        if not isinstance(num_feats, list):
            num_feats = [num_feats] * 3
        self.h_embed = nn.Embedding(h, num_feats[0])
        self.w_embed = nn.Embedding(w, num_feats[1])
        self.num_feats = num_feats
        self.h, self.w = h, w

    def forward(self, bs, device):
        h_embed = self.h_embed(torch.arange(self.h, device=device))
        h_embed = h_embed.reshape(self.h, 1, -1).repeat(1, self.w, 1)
        w_embed = self.w_embed(torch.arange(self.w, device=device))
        w_embed = w_embed.reshape(1, self.w, -1).repeat(self.h, 1, 1)
        z_embed = torch.zeros(
            1, 1, self.num_feats[2],
            device=device).repeat(self.h, self.w, 1)

        pos = torch.cat((h_embed, w_embed, z_embed),
                        dim=-1).flatten(0, 1).unsqueeze(0).repeat(bs, 1, 1)
        return pos