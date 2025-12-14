
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
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim train mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv2_base.py \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train7 \
    --resume /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train7/epoch_20.pth

# todo train脚本位置：/home/lianghao/anaconda3/envs/wangyushentemp/lib/python3.11/site-packages/mmdet3d/.mim/tools/train.py


# todo 评估/可视化
export CUDA_VISIBLE_DEVICES=6
PYTHONPATH=. mim test mmdet3d /home/lianghao/wangyushen/Projects/GaussTR/configs/customs/gausstrv2_base.py \
    -C /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/train6/epoch_24.pth \
    -G 1 \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gausstr/gausstrv2/ours/test