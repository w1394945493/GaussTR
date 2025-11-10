
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