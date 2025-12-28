# 设置进程名
from setproctitle import setproctitle
setproctitle("wys")
import argparse
import os
import os.path as osp
import time
import torch
import numpy as np

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import set_random_seed
from mmdet3d.registry import MODELS,METRICS,HOOKS
from mmengine.optim import build_optim_wrapper
from mmengine.runner.runner import Runner
from mmengine.logging import MMLogger

from timm.scheduler import CosineLRScheduler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def refine_load_from_sd(sd):
    for k in list(sd.keys()):
        # if 'img_neck.' in k or 'lifter.anchor' in k:
        #     del sd[k]
        if 'img_neck.' in k:
            del sd[k]
        if 'lifter.anchor' in k:
            del sd[k]
    return sd

def main(local_rank, args):

    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


    cfg = Config.fromfile(args.py_config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('gaussianformer', log_file=log_file)
    MMLogger._instance_dict['gaussianformer'] = logger
    logger.info('work dir: ' + args.work_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    if args.load_from:
        ckpt = torch.load(args.load_from,map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            model.load_state_dict(refine_load_from_sd(state_dict), strict=False)
        print(f'successfully load from {args.load_from}')


    train_dataloader = Runner.build_dataloader(cfg.train_dataloader)
    val_dataloader = Runner.build_dataloader(cfg.val_dataloader)

    val_evaluator = METRICS.build(cfg.val_evaluator)

    optimizer = build_optim_wrapper(model, cfg.optimizer)
    max_num_epochs = cfg.max_epochs

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataloader) * max_num_epochs,
        lr_min=cfg.optimizer["optimizer"]["lr"] * cfg.get("min_lr_ratio", 0.1),
        warmup_t=cfg.get('warmup_iters', 500),
        warmup_lr_init=1e-6,
        t_in_epochs=False)

    amp = cfg.get('amp', False)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
        os.environ['amp'] = 'true'
    else:
        os.environ['amp'] = 'false'

    epoch = 0
    global_iter = 0
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        model.train()
        os.environ['eval'] = 'false'

        loss_list = []

        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()

        for i_iter, data in enumerate(train_dataloader):

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            data_time_e = time.time()

            with torch.cuda.amp.autocast(amp):
                result_dict = model(imgs=input_imgs, metas=data,mode = 'loss')
                loss = 0.
                for k, v in result_dict.items():
                    loss += v
            if not amp:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm) # todo cfg.grad_max_norm: 35
                optimizer.step()
                optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)

            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:

                base_lr = max([p['lr'] for p in optimizer.param_groups])
                lr = min([p['lr'] for p in optimizer.param_groups])

                logger.info('[TRAIN] Epoch %d Iter %5d/%d: loss: %.3f (%.3f), grad_norm: %.3f, base_lr: %.7f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataloader),
                    loss.item(),
                    np.mean(loss_list),
                    grad_norm,
                    base_lr, lr,
                    time_e - time_s, data_time_e - data_time_s))

                detailed_loss = []
                for loss_name, loss_value in result_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value.item():.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []

            data_time_s = time.time()
            time_s = time.time()
        # save checkpoint
        if local_rank == 0:
            dict_to_save = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'latest.pth')
            torch.save(dict_to_save, save_file_name)

        epoch += 1
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        model.eval()
        os.environ['eval'] = 'true'
        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataloader):
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
                input_imgs = data.pop('img')
                with torch.cuda.amp.autocast(amp):
                    result_dict = model(imgs=input_imgs, metas=data,mode = 'predict')
                    val_evaluator.process(data,result_dict)

        ret_dict = val_evaluator.compute_metrics(results=[])
        txt = ''
        for name,value in ret_dict.items():
            txt += f'{name}: {value} | '
        logger.info(txt)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gaussianformer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--py-config')
    parser.add_argument("--load-from",type=str, default=None)
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    args = parser.parse_args()

    main(0, args)