"""任务队列：单 worker 线程串行执行，状态落盘，重启可恢复。"""

import io
import queue
import secrets
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path

from app.cleanup import IMAGE_SUFFIXES
from app.errors import PipelineError
from app.pipeline import run_pipeline
from app.status import StatusStore, TaskState

_ACTIVE_STATES = {
    TaskState.UPLOADED.value,
    TaskState.PREPROCESSING.value,
    TaskState.COLMAP.value,
    TaskState.TRAINING.value,
    TaskState.MESH_EXPORT.value,
}


class TaskQueue:
    def __init__(self, data_dir: Path, pipeline_fn: Callable[[Path], None] = run_pipeline):
        self.data_dir = Path(data_dir)
        self.tasks_root = self.data_dir / "tasks"
        self.pipeline_fn = pipeline_fn
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_root / task_id

    def create_task(self, zip_bytes: bytes) -> str:
        task_id = secrets.token_hex(3)
        photos_dir = self.task_dir(task_id) / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name  # 拍平路径，防 zip-slip
                if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                with zf.open(info) as src, open(photos_dir / name, "wb") as dst:
                    dst.write(src.read())
                count += 1
        if count == 0:
            raise PipelineError("INSUFFICIENT_IMAGES", "上传的压缩包中没有可识别的图片")
        StatusStore(self.task_dir(task_id)).update(state=TaskState.UPLOADED, progress=0)
        return task_id

    def submit(self, task_id: str) -> None:
        self._queue.put(task_id)

    def start(self) -> None:
        with self._start_lock:
            if self._worker is not None:
                return  # 只允许启动一个 worker，保证串行

            def work() -> None:
                while True:
                    task_id = self._queue.get()
                    try:
                        self.pipeline_fn(self.task_dir(task_id))
                    except Exception as e:  # pipeline_fn 正常应自行落盘 failed，这里是兜底
                        StatusStore(self.task_dir(task_id)).update(
                            state=TaskState.FAILED,
                            error=PipelineError("TRAIN_FAILED", f"内部错误（{type(e).__name__}）"),
                        )
                    finally:
                        self._queue.task_done()

            self._worker = threading.Thread(target=work, daemon=True)
            self._worker.start()

    def recover(self) -> int:
        recovered = 0
        if not self.tasks_root.exists():
            return 0
        for task_dir in sorted(self.tasks_root.iterdir()):
            try:
                status = StatusStore(task_dir).read()
            except FileNotFoundError:
                continue
            if status["state"] in _ACTIVE_STATES:
                self.submit(task_dir.name)
                recovered += 1
        return recovered

    def list_tasks(self) -> list[dict]:
        result = []
        if not self.tasks_root.exists():
            return result
        for task_dir in sorted(self.tasks_root.iterdir()):
            try:
                result.append(StatusStore(task_dir).read())
            except FileNotFoundError:
                continue
        return result
