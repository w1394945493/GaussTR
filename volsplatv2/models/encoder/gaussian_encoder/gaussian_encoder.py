import torch.nn as nn
from mmdet3d.registry import MODELS


@MODELS.register_module()
class GaussianOccEncoder(nn.Module):
    def __init__(self,
                 num_decoder,
                 anchor_encoder,
                 norm_layer,
                 ffn,
                 deformable_layer,
                 spconv_layer,
                 refine_layer,
                 ):
        super().__init__()
        self.num_decoder = num_decoder
        self.anchor_encoder = MODELS.build(anchor_encoder)
        self.norm_layer = MODELS.build(norm_layer)
        self.ffn = MODELS.build(ffn)
        self.deformable_layer = MODELS.build(deformable_layer)
        self.spconv_layer = MODELS.build(spconv_layer)
        self.refine_layer = MODELS.build(refine_layer)
        
        
    def forward(self,
                anchor, # todo (1 25600 3)
                instance_feature, # todo (1 25600 128)
                ms_img_feats,
                projection_mat,featmap_wh): # todo (1 6 4 4) (1 6 2)
        
        predictions = []
        anchor_embed = self.anchor_encoder(anchor) # todo (1 25600 128)
        
        for i in range(self.num_decoder):
            #?-----------------------------?
            instance_feature = self.deformable_layer(instance_feature, anchor, anchor_embed, ms_img_feats,projection_mat,featmap_wh) # todo (1 25600 256)
            #?-----------------------------?
            
            instance_feature = self.ffn(instance_feature) # todo (1 25600 128)
            instance_feature = self.norm_layer(instance_feature) # todo (1 25600 128)
            
            #?-----------------------------?
            instance_feature = self.spconv_layer(instance_feature,anchor) # todo (1 25600 128)
            instance_feature = self.norm_layer(instance_feature)
            
            
            #?-----------------------------?
            # anchor, gaussians = self.refine_layer(instance_feature,anchor,anchor_embed)
            anchor = self.refine_layer(instance_feature,anchor,anchor_embed)
            anchor_embed = self.anchor_encoder(anchor)
            
            # predictions.append(gaussians)
            predictions.append(anchor)
        return predictions
        
        