
from einops import rearrange

from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

from .metrics import compute_psnr,compute_lpips,compute_ssim

@METRICS.register_module()
class EvalMetric(BaseMetric):
    def __init__(self,
        collect_device='cpu',
        prefix=None,
        submission_prefix=None,):

        self.submission_prefix = submission_prefix
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.test_step_outputs = {}

    def process(self, data_batch, data_samples):

        rgb = rearrange(data_samples[0]['img_pred'],'b v c h w -> (b v) c h w')
        rgb_gt = rearrange(data_samples[0]['img_gt'],'b v c h w -> (b v) c h w')

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
        logger: MMLogger = MMLogger.get_current_instance()
        if self.submission_prefix:
            self.format_test_step_outputs(results)
            return None

        ret_dict = dict()
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