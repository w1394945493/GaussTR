# 设置进程名
from setproctitle import setproctitle
setproctitle("wys")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS,DATASETS,DATA_SAMPLERS,METRICS,HOOKS

from gausstr.datasets import collate_fn

def main(args):
    cfg = Config.fromfile(args.py_config)
    # init_default_scope: 初始化默认的注册域
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    model = MODELS.build(cfg.model)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    sampler_dict = cfg.test_dataloader.sampler.copy()
    sampler_dict.update({'dataset':dataset})

    dataset_sampler = DATA_SAMPLERS.build(sampler_dict)
    test_dataloader = DataLoader(
        dataset,
        batch_size=cfg.test_dataloader.batch_size,
        sampler=dataset_sampler,
        num_workers=cfg.test_dataloader.num_workers,
        pin_memory=cfg.test_dataloader.pin_memory,
        drop_last = cfg.test_dataloader.drop_last,
        collate_fn=collate_fn
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint,map_location='cpu')
        model.load_state_dict(ckpt['state_dict'],strict=True)
        print(f"Loaded pretrained weights: {args.checkpoint}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_evaluator=METRICS.build(cfg.test_evaluator)
    test_evaluator.dataset_meta = {
        'label2cat':{i: category for i, category in enumerate(dataset.METAINFO['classes'])}
    }

    dump_result = HOOKS.build(cfg.custom_hooks[0])

    results = []
    with torch.no_grad():
        model.eval()
        for i_iter, (inputs, data_batch) in enumerate(tqdm(test_dataloader, desc="Processing", total=len(test_dataloader))):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(inputs,data_batch,mode='predict')

            data_samples = torch.unbind(outputs, dim=0)
            data_batch = {'data_samples':data_batch}

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