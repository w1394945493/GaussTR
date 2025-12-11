import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from einops import rearrange


from mmdet3d.evaluation import fast_hist, per_class_iou
from mmdet3d.registry import METRICS

from .metrics import compute_psnr,compute_lpips,compute_ssim

def compute_occ_iou(hist, free_index):
    tp = (
        hist[:free_index, :free_index].sum() +
        hist[free_index + 1:, free_index + 1:].sum())
    return tp / (hist.sum() - hist[free_index, free_index])


@METRICS.register_module()
class OccMetric(BaseMetric):

    def __init__(self,
                 num_classes,
                 use_lidar_mask=False,
                 use_image_mask=True,
                 collect_device='cpu',
                 prefix=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask

        self.hist = np.zeros((num_classes, num_classes))
        self.results = []

        self.test_step_outputs = {}

        self.val_occ = False

    def process(self, data_batch, data_samples):

        # todo -------------------------------#
        # todo 占用预测评估
        preds = data_samples[0]['occ_pred'] # (b,X,Y,Z)
        labels = torch.cat(
            [d.gt_pts_seg.semantic_seg for d in data_batch['data_samples']])
        if self.use_image_mask:
            mask = torch.stack([
                torch.from_numpy(d.mask_camera)
                for d in data_batch['data_samples']
            ]).to(labels.device, torch.bool)
        elif self.use_lidar_mask:
            mask = torch.stack([
                torch.from_numpy(d.mask_lidar)
                for d in data_batch['data_samples']
            ]).to(labels.device, torch.bool)
        if self.use_image_mask or self.use_lidar_mask:
            preds = preds[mask]
            labels = labels[mask]

        preds = preds.flatten().cpu().numpy()
        labels = labels.flatten().cpu().numpy()
        hist_ = fast_hist(preds, labels, self.num_classes) # 计算混淆矩阵
        self.hist += hist_

        # todo -------------------------------#
        # todo 视图合成评估
        rgb = rearrange(data_samples[0]['img_pred'],'b v c h w -> (b v) c h w')
        # rgb_gt = rearrange(data_samples[0]['img_gt'],'b v c h w -> (b v) c h w')
        rgb_gt = torch.cat(
            [d.img for d in data_batch['data_samples']],dim=0) / 255.
        rgb_gt = rgb_gt.to(rgb.device)

        if f"psnr" not in self.test_step_outputs:
            self.test_step_outputs[f"psnr"] = []
        if f"ssim" not in self.test_step_outputs:
            self.test_step_outputs[f"ssim"] = []
        if f"lpips" not in self.test_step_outputs:
            self.test_step_outputs[f"lpips"] = []

        self.test_step_outputs[f"psnr"].append(
            compute_psnr(rgb_gt, rgb).mean().item()
        )
        self.test_step_outputs[f"ssim"].append(
            compute_ssim(rgb_gt, rgb).mean().item()
        )
        self.test_step_outputs[f"lpips"].append(
            compute_lpips(rgb_gt, rgb).mean().item()
        )




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

        # todo 评估结果记录
        ret_dict = dict()

        # todo -----------------------------#
        # todo occ占用预测评估结果打印
        iou = per_class_iou(self.hist)
        # if ignore_index is in iou, replace it with nan
        miou = np.nanmean(iou[:-1])  # NOTE: ignore free class
        label2cat = self.dataset_meta['label2cat']

        header = ['classes']
        for i in range(len(label2cat) - 1):
            header.append(label2cat[i])
        header.extend(['miou', 'iou'])
        table_columns = [['results']]
        for i in range(len(label2cat) - 1):
            ret_dict[label2cat[i]] = float(iou[i])
            table_columns.append([f'{iou[i]:.4f}'])
        ret_dict['miou'] = float(miou)
        ret_dict['iou'] = compute_occ_iou(self.hist, self.num_classes - 1)
        table_columns.append([f'{miou:.4f}'])
        table_columns.append([f"{ret_dict['iou']:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        # todo -----------------------------#
        # todo 视图合成评估结果打印
        header = ['metric name']
        table_columns = ['results']
        for metric_name, metric_scores in self.test_step_outputs.items():
            avg_scores = sum(metric_scores) / len(metric_scores)
            ret_dict[metric_name] = avg_scores
            header.append(metric_name)
            table_columns.append(f'{avg_scores:.4f}')

        table_data = [header,table_columns]
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

        return ret_dict


