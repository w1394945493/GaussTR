import numpy as np
import torch
import torch.distributed as dist
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from einops import rearrange


from mmdet3d.evaluation import fast_hist, per_class_iou
from mmdet3d.registry import METRICS

from .img_metrics import compute_psnr,compute_lpips,compute_ssim

def compute_occ_iou(hist, free_index):
    tp = (
        hist[:free_index, :free_index].sum() +
        hist[free_index + 1:, free_index + 1:].sum())
    return tp / (hist.sum() - hist[free_index, free_index])


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

        self.reset()

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()

    def process(self, data_batch, data_samples):
        preds = data_samples[0]['occ_pred'] # (b,(h w d))
        gt_occ = data_samples[0]['occ_gt']
        occ_mask = data_samples[0]['occ_mask']
        for idx in range(preds.size(0)):
            outputs = preds[idx]
            targets = gt_occ[idx]
            mask = occ_mask[idx]
            outputs = outputs[mask]
            targets = targets[mask]
            for i, c in enumerate(self.class_indices):
                self.total_seen[i] += torch.sum(targets == c).item()
                self.total_correct[i] += torch.sum((targets == c)
                                                & (outputs == c)).item()
                self.total_positive[i] += torch.sum(outputs == c).item()

            self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
            self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                                & (outputs != self.empty_label)).item()
            self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def compute_metrics(self, results):

        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        # todo 评估结果记录
        ret_dict = dict()

        ious = []
        precs = []
        recas = []

        for i in range(self.num_classes):
            if self.total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(cur_prec.item())
            if self.total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)
        miou = np.mean(ious)
        occ_iou = self.total_correct[-1] / (self.total_seen[-1] + self.total_positive[-1] - self.total_correct[-1])

        for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
            print_log('%s : %.2f%%, %.2f, %.2f' % (label_str, iou * 100, prec, reca),logger=logger)

        print_log(self.total_seen.int(),logger=logger)
        print_log(self.total_correct.int(),logger=logger)
        print_log(self.total_positive.int(),logger=logger)

        ret_dict['miou'] = miou * 100
        ret_dict['iou'] = occ_iou * 100

        return ret_dict


