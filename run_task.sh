PYTHONPATH=. python -m torch.distributed.launch \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --node_rank=$MLP_ROLE_INDEX \
    --master_port=$MLP_WORKER_0_PORT \
    --nnodes=$MLP_WORKER_NUM \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv3_experiment.py \
    --launcher pytorch \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/train \
    --resume /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/train/epoch_3.pth