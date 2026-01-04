import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from einops import rearrange

from mmdet3d.registry import METRICS
from .metrics import compute_psnr,compute_lpips,compute_ssim



@METRICS.register_module()
class ImgMetric(BaseMetric):

    def __init__(self,
                 collect_device='cpu',
                 prefix=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super().__init__(prefix=prefix, collect_device=collect_device)

        self.results = []

        self.test_step_outputs = {}

        self.occ_flag = False

    def process(self, data_batch, data_samples):

        rgb = rearrange(data_samples[0]['img_pred'],'b v c h w -> (b v) c h w')
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
        
        self.test_step_outputs = {}
        return ret_dict


