"""测试全局配置。

在任何测试模块 import src.graph 之前，把 checkpoint db 指向一次性临时文件，
避免测试污染真实的 .checkpoints/checkpoints.sqlite。
"""
import os
import tempfile

os.environ.setdefault(
    "CHECKPOINT_DB_PATH",
    tempfile.mktemp(prefix="dra_ckpt_", suffix=".sqlite"),
)
