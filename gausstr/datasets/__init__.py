from .nuscenes_occ import NuScenesOccDataset
from .transforms import *




def collate_fn(batch):
    inputs_list = []
    data_samples_list = []
    # 遍历 batch 中的每一个元素
    for sample in batch:
        inputs_list.append(sample['inputs'])
        data_samples_list.append(sample['data_samples'])

    inputs = {}
    for key in inputs_list[0].keys():  # 假设每个 sample 的 inputs 字典结构相同
        values = [input_dict[key] for input_dict in inputs_list]
        if isinstance(values[0], torch.Tensor):
            inputs['imgs'] = torch.stack(values)
        else:
            inputs['imgs'] = torch.from_numpy(np.stack(values))

    return inputs, data_samples_list