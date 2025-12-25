
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@HOOKS.register_module()
class DumpResultHook(Hook):

    def before_val_epoch(self, runner, **kwargs):
        print(cyan(f'occ metrics reset!'))
        runner.val_evaluator.metrics[0].reset()

    def before_test_epoch(self, runner,**kwargs):
        print(cyan(f'occ metrics reset!'))
        runner.val_evaluator.metrics[0].reset()


