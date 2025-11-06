# 设置进程名
from setproctitle import setproctitle
setproctitle("wys")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse


import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg) # 加载配置项，调用构造函数
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test() # dataloader等构建，在此实现


if __name__ == '__main__':
    main()
