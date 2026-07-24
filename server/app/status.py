"""任务状态存储。status.json 是任务状态的唯一来源（无数据库）。"""

import json
import os
from enum import Enum
from pathlib import Path

from app.errors import PipelineError


class TaskState(str, Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    COLMAP = "colmap"
    TRAINING = "training"
    MESH_EXPORT = "mesh_export"
    DONE = "done"
    FAILED = "failed"


_INITIAL = {
    "state": TaskState.UPLOADED.value,
    "progress": 0,
    "stage_detail": "",
    "eta_seconds": None,
    "error": None,
}


class StatusStore:
    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir)
        self.path = self.task_dir / "status.json"

    def read(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            status = json.load(f)
        status["task_id"] = self.task_dir.name
        return status

    def update(
        self,
        state: TaskState | None = None,
        progress: int | None = None,
        stage_detail: str | None = None,
        error: PipelineError | None = None,
        eta_seconds: int | None = None,
    ) -> None:
        status = dict(_INITIAL)
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                status.update(json.load(f))
        if state is not None:
            status["state"] = state.value
        if progress is not None:
            status["progress"] = progress
        if stage_detail is not None:
            status["stage_detail"] = stage_detail
        if eta_seconds is not None:
            status["eta_seconds"] = eta_seconds
        if error is not None:
            status["error"] = {"code": error.code, "message_zh": error.message_zh}
        # 原子写：先写临时文件再 rename，避免读到写了一半的 JSON
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)
