import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue


def _zip_bytes(names=("a.jpg",)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"fake-image")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    q = TaskQueue(tmp_path, pipeline_fn=lambda d: None)
    app = create_app(tmp_path, queue=q)
    return TestClient(app), q


def test_create_task(client):
    c, q = client
    resp = c.post("/api/tasks", files={"file": ("photos.zip", _zip_bytes(), "application/zip")})
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]
    status = c.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "uploaded"


def test_create_task_rejects_empty_zip(client):
    c, _ = client
    resp = c.post("/api/tasks", files={"file": ("x.zip", _zip_bytes(names=("a.txt",)), "application/zip")})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INSUFFICIENT_IMAGES"


def test_status_404(client):
    c, _ = client
    assert c.get("/api/tasks/zzzzzz/status").status_code == 404


def test_model_download_requires_done(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.get(f"/api/tasks/{task_id}/model").status_code == 409
    task_dir = q.task_dir(task_id)
    StatusStore(task_dir).update(state=TaskState.DONE, progress=100)
    (task_dir / "model.glb").write_bytes(b"glb-bytes")
    resp = c.get(f"/api/tasks/{task_id}/model")
    assert resp.status_code == 200
    assert resp.content == b"glb-bytes"
    assert resp.headers["content-type"] == "model/gltf-binary"


def test_thumbnail(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.get(f"/api/tasks/{task_id}/thumbnail").status_code == 404
    (q.task_dir(task_id) / "thumbnail.jpg").write_bytes(b"jpeg")
    assert c.get(f"/api/tasks/{task_id}/thumbnail").status_code == 200


def test_delete_task(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.delete(f"/api/tasks/{task_id}").status_code == 409  # uploaded 属活跃态
    StatusStore(q.task_dir(task_id)).update(state=TaskState.DONE, progress=100)
    assert c.delete(f"/api/tasks/{task_id}").status_code == 204
    assert c.get(f"/api/tasks/{task_id}/status").status_code == 404
