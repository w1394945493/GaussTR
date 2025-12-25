import math
from typing import Callable, List, Optional, Sequence, Union
from torch.optim import Optimizer

from mmengine.optim import BaseOptimWrapper
from mmengine.registry import PARAM_SCHEDULERS

from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim import _ParamScheduler

INF = int(1e9)

@PARAM_SCHEDULERS.register_module()
class CosineLRScheduler(_ParamScheduler):
    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,

                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 cycle_limit=0,
                 warmup_t: int = 500,
                 warmup_lr_init: float= 1e-6,

                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit

        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.total_iters = end - begin - 1

        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
        else:
            self.warmup_steps = [1 for _ in self.base_values]


    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        t = self.last_step
        if t == 0:
            if self.warmup_t:
                lrs = [self.warmup_lr_init] * len(self.optimizer.param_groups)
            else:
                lrs = [group[self.param_name] for group in self.optimizer.param_groups]

        elif t <= self.warmup_t:
            # todo 预热阶段
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # todo 余弦退火
            i = t // self.total_iters
            t_i = self.total_iters
            t_curr = t - (self.total_iters * i)
            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

@PARAM_SCHEDULERS.register_module()
class CosineLR(LRSchedulerMixin, CosineLRScheduler):
    pass


