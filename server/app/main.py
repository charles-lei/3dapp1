"""FastAPI 服务入口。启动：python -m app.main（默认 0.0.0.0:8000）。"""

import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.errors import PipelineError
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_TERMINAL_STATES = {TaskState.DONE.value, TaskState.FAILED.value}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task_queue: TaskQueue = app.state.queue
    recovered = task_queue.recover()
    task_queue.start()
    if recovered:
        print(f"恢复 {recovered} 个中断任务")
    yield


def create_app(data_dir: Path = DEFAULT_DATA_DIR, queue: TaskQueue | None = None) -> FastAPI:
    task_queue = queue or TaskQueue(data_dir)
    app = FastAPI(title="3D Recon Server", lifespan=_lifespan)
    app.state.queue = task_queue

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request, exc: PipelineError):
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message_zh": exc.message_zh}},
        )

    @app.post("/api/tasks", status_code=201)
    async def create_task(file: UploadFile):
        task_id = task_queue.create_task(await file.read())
        task_queue.submit(task_id)
        return {"task_id": task_id}

    @app.get("/api/tasks/{task_id}/status")
    async def get_status(task_id: str):
        try:
            return StatusStore(task_queue.task_dir(task_id)).read()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="task not found")

    @app.get("/api/tasks/{task_id}/model")
    async def get_model(task_id: str):
        status = _read_status_or_404(task_queue, task_id)
        if status["state"] != TaskState.DONE.value:
            raise HTTPException(status_code=409, detail="model not ready")
        return FileResponse(
            task_queue.task_dir(task_id) / "model.glb",
            media_type="model/gltf-binary",
            filename="model.glb",
        )

    @app.get("/api/tasks/{task_id}/thumbnail")
    async def get_thumbnail(task_id: str):
        path = task_queue.task_dir(task_id) / "thumbnail.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="thumbnail not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.delete("/api/tasks/{task_id}", status_code=204)
    async def delete_task(task_id: str):
        status = _read_status_or_404(task_queue, task_id)
        if status["state"] not in _TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="task still running")
        shutil.rmtree(task_queue.task_dir(task_id))

    return app


def _read_status_or_404(task_queue: TaskQueue, task_id: str) -> dict:
    try:
        return StatusStore(task_queue.task_dir(task_id)).read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="task not found")


def _lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


app = create_app()

if __name__ == "__main__":
    print(f"局域网地址: http://{_lan_ip()}:8000  （在 App 设置中填写此地址）")
    uvicorn.run(app, host="0.0.0.0", port=8000)
