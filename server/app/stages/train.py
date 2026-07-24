"""阶段 3：3DGS 训练（ns-train splatfacto 封装）。"""

import re
from collections.abc import Callable
from pathlib import Path

from app.errors import CommandError, PipelineError
from app.interfaces import ProgressCallback
from app.runner import run_command


def parse_iteration(line: str, max_iterations: int) -> int | None:
    match = re.search(rf"\b(\d+)\s*/\s*{max_iterations}\b", line)
    return int(match.group(1)) if match else None


def is_oom(output: str) -> bool:
    return "CUDA out of memory" in output or "OutOfMemoryError" in output


def find_config(train_output_dir: Path) -> Path:
    configs = sorted(train_output_dir.glob("splatfacto/*/config.yml"))
    if not configs:
        raise PipelineError("TRAIN_FAILED", "训练结束但未找到 config.yml 产物")
    return configs[-1]


class SplatfactoTrainer:
    def __init__(self, max_iterations: int = 10000, runner: Callable = run_command):
        self.max_iterations = max_iterations
        self.runner = runner

    def train(self, task_dir: Path, on_progress: ProgressCallback) -> Path:
        cmd = [
            "ns-train", "splatfacto",
            "--data", str(task_dir / "colmap"),
            "--output-dir", str(task_dir / "train_output"),
            "--max-num-iterations", str(self.max_iterations),
            "--vis", "tensorboard",
        ]
        last_reported = -1

        def on_line(line: str) -> None:
            nonlocal last_reported
            iteration = parse_iteration(line, self.max_iterations)
            if iteration is not None:
                pct = round(iteration / self.max_iterations * 100)
                if pct != last_reported:
                    last_reported = pct
                    on_progress(pct, f"iteration {iteration}/{self.max_iterations}")

        try:
            self.runner(cmd, on_line=on_line)
        except CommandError as e:
            if is_oom(e.output):
                raise PipelineError(
                    "TRAIN_OOM", "显存不足（CUDA OOM），将自动降分辨率重试"
                ) from e
            raise PipelineError(
                "TRAIN_FAILED", f"训练进程异常退出（exit {e.returncode}），请重试或联系开发者"
            ) from e
        return find_config(task_dir / "train_output")
