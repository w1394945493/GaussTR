寻找 段错误（核心已转储）的原因：
1.开启限制：在终端输入 ulimit -c unlimited（仅对当前终端有效）
2.export CUDA_LAUNCH_BLOCKING=1
3.在train.py 或 main.py 最开头添加
import faulthandler
faulthandler.enable()

报错：
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
