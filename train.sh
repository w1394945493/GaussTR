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


conda deactivate
conda create --prefix /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp python=3.10 pip openssl -y
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
conda install pip --force-reinstall -y # !!! 明确告诉它装到你的新环境路径下
# 检查
which pip 
python -c "import ssl; print('SSL OK')"
python -c "import platform; print(platform.python_implementation())"
# 
pip install /c20250502/wangyushen/whl/torch-2.1.1+cu121-cp310-cp310-linux_x86_64.whl
pip install /c20250502/wangyushen/whl/torchvision-0.16.1+cu121-cp310-cp310-linux_x86_64.whl

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html --no-cache-dir

pip install -r /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR_copy/requirements.txt
pip install numpy==1.23.5
# 安装minkowskiengine
conda install -c conda-forge openblas python=3.10 -y
python setup.py install --blas=openblas

# 编译自定义包时，避免环境隔离：--no-build-isolation
pip install -e . --no-build-isolation
pip install . --no-build-isolation # 放弃可编辑模式
python setup.py install
rm -rf build/ dist/ *.egg-info

# MinkowskiEngine在cuda 12.8下编译
git clone https://github.com/chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda-12-1
# Uncomment the following line if you are using CUDA 12.8.
sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' /usr/include/c++/11/bits/shared_ptr_base.h
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas


# todo ---------------------------------------------------------#
# todo volsplatv2 使用整个nuscenes数据集(1/10)进行训练
export PYTHONPATH=$PYTHONPATH:$(pwd)

# todo 单卡训练
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6

python /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/train.py \
    /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv2_experiment.py \
    --work-dir /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/outputs/gausstr/volsplatv2/train \
    --resume /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/outputs/gausstr/volsplatv2/train/epoch_16.pth



PYTHONPATH=. torchrun --nproc_per_node=2 \
    /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/train.py \
    /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv2_experiment.py \
    --launcher pytorch \
    --load-from /vepfs-mlp2/mlp-public/haoce/wangyushen/GaussTR/outputs/gausstr/volsplatv2/train/epoch_2.pth \
    --work-dir /c20250502/wangyushen/Outputs/gausstr/volsplatv2/train

# todo 评估
export CUDA_VISIBLE_DEVICES=6
python /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv2_experiment.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv2/test \
    --checkpoint /c20250502/wangyushen/Outputs/gausstr/volsplatv2/train/epoch_24.pth \

# volsplatv3 train
export CUDA_VISIBLE_DEVICES=0
python /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv3_experiment.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/train \

python /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/GaussTR/configs/volsplat/volsplatv3_experiment.py \
    --work-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/test \
    --checkpoint /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gausstr/volsplatv3/train/epoch_24.pth \





