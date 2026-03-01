#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="local_aggregate", # python包的名字
    packages=['local_aggregate'], # 告诉python哪些文件夹应该被视为python包
    ext_modules=[
        CUDAExtension(  # CUDAExtension(): 定义了如何编译底层代码
            name="local_aggregate._C", # 编译出来的二进制文件名：通过from . import _C来调用
            sources=[                  # 列出所有需要编译的源文件，包括实现逻辑的.cu文件和将C++接口暴露给Pytorch的ext.cpp
            "src/aggregator_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "local_aggregate.cu",
            "ext.cpp"],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-g", "-G", "-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension # 告诉python使用Pytorch提供的编译器
    }
)