import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS


torch.autograd.set_detect_anomaly(True)

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"



@MODELS.register_module()
class DepthSplat(BaseModel):
    def __init__(self,
                 backbone,
                 transformer,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.backbone = MODELS.build(backbone)
        self.transformer = MODELS
        
        
        
        return
    
    def forward(self, mode='loss',**data):
        
        
        return