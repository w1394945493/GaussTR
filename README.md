# gaussianformer
# 数据集：使用surroundocc
# 评估：
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train/epoch_16.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/test

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Weights/gaussianformer/custom/state_dict.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/test

# 训练：
export CUDA_VISIBLE_DEVICES=3
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train

