import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from einops import rearrange

from mmdet3d.registry import METRICS

@METRICS.register_module()
class OccMetric(BaseMetric):

    def __init__(self,
                 class_indices,
                 empty_label,
                 label_str,
                 dataset_empty_label=17,
                 filter_minmax=True,                 
                 collect_device='cpu',
                 prefix=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.results = []
        super().__init__(prefix=prefix, collect_device=collect_device)

        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.filter_minmax = filter_minmax
        
        self.test_step_outputs = {}

        self.reset()

    def reset(self):
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()

    def process(self, data_batch, data_samples):

        
        occ_pred, occ_mask, occ_gt = data_samples[0]['occ_pred'], data_samples[0]['occ_mask'],data_samples[0]['occ_gt'] # (b,200,200,16)
        occ_pred = occ_pred.flatten(1) # (b,(h w d))
        occ_mask = occ_mask.flatten(1)
        occ_gt = occ_gt.flatten(1)
        for idx in range(occ_pred.size(0)):
            outputs = occ_pred[idx]
            targets = occ_gt[idx]
            mask = occ_mask[idx]
            outputs = outputs[mask]
            targets = targets[mask]        
            for i, c in enumerate(self.class_indices):
                self.total_seen[i] += torch.sum(targets == c).item() # todo GT中某类的voxel数(TP+FN)
                self.total_correct[i] += torch.sum((targets == c)
                                                & (outputs == c)).item() # todo 预测对的voxel数(TP)
                self.total_positive[i] += torch.sum(outputs == c).item() # todo 预测为某类的voxel数(TP+FP)
            # todo 整体occupancy(iou)
            self.total_seen[-1] += torch.sum(targets != self.empty_label).item()  # todo GT中非空类的voxel数(TP+FN)
            self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                                & (outputs != self.empty_label)).item() # todo 预测对的非空数(TP)
            self.total_positive[-1] += torch.sum(outputs != self.empty_label).item() # todo 预测为非空类的数(TP+FP)
                
        

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        torch.cuda.synchronize()
        
        
        # todo 评估结果记录
        ret_dict = dict()


        # todo -----------------------------#
        # todo occ占用预测评估结果打印
        total_seen = self.total_seen.cpu().numpy()
        total_correct = self.total_correct.cpu().numpy()
        total_positive = self.total_positive.cpu().numpy()
        ious = []
        header = ['classes']
        for i in range(len(self.label_str)):
            header.append(self.label_str[i])
        header.extend(['miou', 'iou'])
        table_columns = [['results']]

        for i in range(self.num_classes): # todo 只计算语义类，不包括非空类
            if self.total_seen[i] == 0: # todo iou & recall
                cur_iou = np.nan
            else:
                cur_iou = total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i]) # todo iou = TP / (TP + FN + FP)

            ious.append(cur_iou)
            table_columns.append([f'{cur_iou:.4f}'])

            ret_dict[self.label_str[i]] = cur_iou * 100

        miou = np.nanmean(ious)
        iou = total_correct[-1] / (total_seen[-1] + total_positive[-1] - total_correct[-1])

        table_columns.append([f'{miou:.4f}'])
        table_columns.append([f"{iou:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        ret_dict['miou'] = miou * 100
        ret_dict['iou'] = iou * 100
        self.reset()

        return ret_dict


