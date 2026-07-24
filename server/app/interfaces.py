"""可替换阶段接口。换 SuGaR 等方案时只新增实现类，不动编排器。"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ProgressCallback = Callable[[int, str], None]  # (阶段内进度 0~100, 描述文本)


class Trainer(Protocol):
    def train(self, task_dir: Path, on_progress: ProgressCallback) -> Path:
        """训练并返回产物 config.yml 路径。"""
        ...


class MeshExporter(Protocol):
    def export(self, task_dir: Path, config_path: Path, on_progress: ProgressCallback) -> Path:
        """导出并返回原始 mesh .ply 路径。"""
        ...
