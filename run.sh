
pip install git+https://github.com/mhamilton723/FeatUp

# todo match the occupancy ground truths:
python tools/update_data.py \
    nuscenes \
    --root-path /home/lianghao/wangyushen/data/wangyushen/Datasets/nuscenes/v1.0-mini \
    --version v1.0-mini \
    --out-dir /home/lianghao/wangyushen/data/wangyushen/Datasets/nuscenes/v1.0-mini \
    --extra-tag nuscenes_mini



# todo Run to generate metric depth estimations
# todo 对nuscenes数据集的pkl文件进行深度预测，将预测结果保存为.npy格式文件
python tools/generate_depth.py
python /home/lianghao/wangyushen/Projects/GaussTR/tools/generate_depth.py

# todo 使用talk2dino model：
PYTHONPATH=. mim test mmdet3d [CONFIG] -C [CKPT_PATH] [-l pytorch -G [GPU_NUM]]

# todo PYTHONPATH=.:设置了环境变量,在当前目录下寻找模块和包,确保自定义模块可以被正确导入
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim test mmdet3d configs/customs/gausstr_talk2dino.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Weights/gausstr/gausstr_talk2dino_e20_miou12.27.pth \
    -l pytorch -G 1

# todo 参考 mmdet3d.mim/tools/test.py
python /home/lianghao/wangyushen/Projects/GaussTR/test.py \
    configs/customs/gausstr_talk2dino.py \
    /home/lianghao/wangyushen/data/wangyushen/Weights/gausstr/gausstr_talk2dino_e20_miou12.27.pth \
    --launcher pytorch \


# todo 可视化
python tools/visualize.py [PKL_PATH] [--save]

export QT_QPA_PLATFORM=offscreen
python tools/visualize.py \
    /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/test_debug/vis \
    --save

# todo demo 推理 与 终端 PYTHONPATH=. mim test mmdet3d 命令运行结果不一致, 待寻找原因: 输入图像缺少归一化操作
python /home/lianghao/wangyushen/Projects/GaussTR/demo.py \
    --py-config \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstr_talk2dino.py \
    --checkpoint \
    /home/lianghao/wangyushen/data/wangyushen/Weights/gausstr/gausstr_talk2dino_e20_miou12.27.pth \
    --vis_result \

# todo GaussTR train
PYTHONPATH=. mim train mmdet3d [CONFIG] [-l pytorch -G [GPU_NUM]]

# todo 参考 mmdet3d.mim/tools/trai.py
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim train mmdet3d configs/customs/gausstr_talk2dino.py \
    -l pytorch -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/train_debug

# todo 生成语义分割图
python /home/lianghao/wangyushen/Projects/GaussTR/tools/generate_grounded_sam2.py

python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstr_v2_talk2dino.py \
    --work-dir \
    /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/train_debug

# todo 训练
# todo baseline
export CUDA_VISIBLE_DEVICES=4
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstr_talk2dino.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/baseline/train

# todo 评估/可视化
export CUDA_VISIBLE_DEVICES=4
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstr_talk2dino.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Weights/gausstr/gausstr_talk2dino_e20_miou12.27.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/baseline/test

python /home/lianghao/wangyushen/Projects/GaussTR/demo.py \
    --py-config \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstr_talk2dino.py \
    --checkpoint \
    /home/lianghao/wangyushen/data/wangyushen/Weights/gausstr/gausstr_talk2dino_e20_miou12.27.pth \



# todo -------------------------------------#
# todo 重新整理MonoSplat代码
# 训练
export CUDA_VISIBLE_DEVICES=5
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/monosplat_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/monosplat/ours/train6 \

# --resume

# 评估
export CUDA_VISIBLE_DEVICES=4
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/monosplat_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/monosplat/ours/train6/epoch_24.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/monosplat/ours/test

PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/monosplat_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train/epoch_24.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/monosplat/ours/test

# todo GaussTRV2
# todo 训练
export CUDA_VISIBLE_DEVICES=4
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv2_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train8 \
    --resume /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train7/epoch_20.pth

# todo train脚本位置：/home/lianghao/anaconda3/envs/wangyushentemp/lib/python3.11/site-packages/mmdet3d/.mim/tools/train.py


# todo 评估/可视化
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv2_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train6/epoch_24.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/test

export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv3_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv3/ours/train \


export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv4_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv4/ours/train

# todo  gaussianformer
# todo 评估：
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

export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/test \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train/epoch_24.pth \


export CUDA_VISIBLE_DEVICES=0
python /home/lianghao/wangyushen/Projects/GaussTR/test.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    --checkpoint /home/lianghao/wangyushen/data/wangyushen/Weights/gaussianformer/custom/state_dict.pth \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/test \


# !---------------------------------------------------------------------------#
# todo 训练：
# todo --resume: 恢复训练
# todo --load-from: 仅加载权重
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -G 2 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train2 \
    --load-from /home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/r101_dcn_fcos3d_pretrain.pth \


PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train \
    --load-from /home/lianghao/wangyushen/data/wangyushen/Weights/gaussianformer/custom/state_dict.pth \

# todo -------------------------------------------------------
export CUDA_VISIBLE_DEVICES=5
python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train2 \
    --load-from /home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/r101_dcn_fcos3d_pretrain.pth \


export CUDA_VISIBLE_DEVICES=0
python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformer/gaussianformer_base.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformer/train \
    --load-from /home/lianghao/wangyushen/data/wangyushen/Weights/gaussianformer/custom/state_dict.pth \


# !---------------------------------------------------------------------------#
# todo gaussianformerv2
# todo 训练：
# ! 使用mmdet.ResNet101作为主干时，记着加载预训练权重！！！
export CUDA_VISIBLE_DEVICES=0
python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformerv2/gaussianformer_base.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/train3 \
    --load-from /home/lianghao/wangyushen/data/wangyushen/Weights/pretrained/r101_dcn_fcos3d_pretrain.pth \

export CUDA_VISIBLE_DEVICES=2
python /home/lianghao/wangyushen/Projects/GaussTR/test.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/gaussianformerv2/gaussianformer_base.py \
    --checkpoint /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/train/epoch_24.pth \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/test \


# todo --------------------------------------------------------#
# todo VolSplat
# todo 训练
export CUDA_VISIBLE_DEVICES=1
python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/volsplat_base.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplat/train


python /home/lianghao/wangyushen/Projects/GaussTR/test.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/volsplat_base.py \
    --checkpoint /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplat/train/best_psnr_epoch_4.pth \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/test \

# todo --------------------------------------------------------#
# todo VolSplatv2
# todo 训练
export CUDA_VISIBLE_DEVICES=5
python /home/lianghao/wangyushen/Projects/GaussTR/train.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/volsplat/volsplatv2_base.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplatv2/train3


python /home/lianghao/wangyushen/Projects/GaussTR/test.py \
    /home/lianghao/wangyushen/Projects/GaussTR/configs/volsplat/volsplatv2_base.py \
    --checkpoint /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/volsplatv2/train3/epoch_24.pth \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gaussianformerv2/test \

# todo ---------------------------------------------------------#
# todo 整理整个nuscenes数据集
ln -s /home/A_DataSets_01/Nuscenes/v1.0-trainval /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval
ln -s /home/A_DataSets_01/Nuscenes/samples/ /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval

# todo 2026.01.13 wys
# todo 从GaussTR下载pkl文件
python tools/update_data.py \
    nuscenes \
    --root-path /home/A_DataSets_01/Nuscenes \
    --version v1.0 \
    --out-dir /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-trainval \
    --extra-tag nuscenes \

# todo 生成深度图
python /home/lianghao/wangyushen/Projects/GaussTR/tools/generate_depth.py