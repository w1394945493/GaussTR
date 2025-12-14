
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class CustomHook(Hook):
    def __init__(self,val_occ_epoch=100):
        self.val_occ_epoch = val_occ_epoch

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch # todo runner.epoch: 从0开始
        if current_epoch >= self.val_occ_epoch:
            runner.model.gauss_head.occ_flag = True
            runner.val_evaluator.metrics[0].occ_flag = True
        return

    def before_test(self, runner):
        runner.model.gauss_head.occ_flag = True
        runner.test_evaluator.metrics[0].occ_flag = True
        return



