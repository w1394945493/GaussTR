# 设置进程名
from setproctitle import setproctitle
setproctitle("wys")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS,METRICS,HOOKS

from mmengine.runner.runner import Runner
from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from gausstr import *
from gausstrv2 import *

def main(args):
    cfg = Config.fromfile(args.py_config)
    # init_default_scope: 初始化默认的注册域
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS.build(cfg.model)
    model.to(device)

    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # if args.checkpoint:
    #     ckpt = torch.load(args.checkpoint,map_location='cpu')
    #     model.load_state_dict(ckpt['state_dict'],strict=True)
    #     print(f"Loaded pretrained weights: {args.checkpoint}")

    test_evaluator=METRICS.build(cfg.test_evaluator)

    if hasattr(test_dataloader.dataset, 'metainfo'):
        test_evaluator.dataset_meta = test_dataloader.dataset.metainfo

    dump_result = HOOKS.build(cfg.custom_hooks[0])

    results = []

    with torch.no_grad():
        model.eval()
        for i_iter, data_batch in enumerate(tqdm(test_dataloader, desc="Processing", total=len(test_dataloader))):

            outputs = model.test_step(data_batch) # data_processer: 归一化

            data_samples = []
            for output in outputs:
                data_samples.append(output)

            test_evaluator.process(data_batch=data_batch,
                                   data_samples=data_samples)
            if args.vis_result:
                dump_result.after_test_iter(runner=None,
                                            batch_idx=i_iter,
                                            data_batch = data_batch,
                                            outputs = outputs,
                                            )
        results=test_evaluator.compute_metrics(results)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config')
    parser.add_argument("--checkpoint",type=str, default=None)
    parser.add_argument("--vis_result",action="store_true")
    args = parser.parse_args()

    main(args)