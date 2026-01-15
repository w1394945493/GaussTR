from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gauss_splatting_cuda',
    ext_modules=[
        CUDAExtension('gauss_splatting_cuda', [
            'splatting_cuda.cpp',
            'splatting_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)