import numpy as np
import torch
from mmengine.registry import FUNCTIONS

@FUNCTIONS.register_module()
def custom_collate_fn_temporal(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if isinstance(v, np.ndarray):
            return_dict[k] = torch.stack([
                torch.from_numpy(instance[k]) for instance in instances])
        elif isinstance(v, torch.Tensor):
            return_dict[k] = torch.stack([instance[k] for instance in instances])
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            raise NotImplementedError
    return return_dict