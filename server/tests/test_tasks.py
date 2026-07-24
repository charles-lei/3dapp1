import io
import json
import time
import zipfile

import pytest

from app.errors import PipelineError
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue


def _zip_bytes(names=("a.jpg", "b.jpg")):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"fake-image-bytes")
    return buf.getvalue()


def test_create_task_extracts_zip_and_inits_status(tmp_path):
    q = TaskQueue(tmp_path)
    task_id = q.create_task(_zip_bytes())
    assert len(task_id) == 6
    task_dir = q.task_dir(task_id)
    assert sorted(p.name for p in (task_dir / "photos").iterdir()) == ["a.jpg", "b.jpg"]
    status = StatusStore(task_dir).read()
    assert status["state"] == "uploaded"


def test_create_task_rejects_non_image_entries(tmp_path):
    q = TaskQueue(tmp_path)
    with pytest.raises(PipelineError):
        q.create_task(_zip_bytes(names=("evil.exe",)))


def test_worker_executes_tasks_serially(tmp_path):
    events = []

    def fake_pipeline(task_dir):
        events.append(("start", task_dir.name))
        time.sleep(0.05)
        events.append(("end", task_dir.name))

    q = TaskQueue(tmp_path, pipeline_fn=fake_pipeline)
    ids = [q.create_task(_zip_bytes()) for _ in range(2)]
    for i in ids:
        q.submit(i)
    q.start()
    deadline = time.time() + 5
    while len(events) < 4 and time.time() < deadline:
        time.sleep(0.01)
    # 串行：第一个任务 end 之后才有第二个任务的 start
    assert events == [
        ("start", ids[0]), ("end", ids[0]),
        ("start", ids[1]), ("end", ids[1]),
    ]


def test_recover_resubmits_interrupted_tasks(tmp_path):
    q = TaskQueue(tmp_path, pipeline_fn=lambda d: None)
    done_id = q.create_task(_zip_bytes())
    StatusStore(q.task_dir(done_id)).update(state=TaskState.DONE, progress=100)
    stuck_id = q.create_task(_zip_bytes())
    StatusStore(q.task_dir(stuck_id)).update(state=TaskState.TRAINING, progress=50)
    assert q.recover() == 1  # 只有 training 的被恢复


def test_list_tasks(tmp_path):
    q = TaskQueue(tmp_path)
    ids = [q.create_task(_zip_bytes()) for _ in range(2)]
    tasks = q.list_tasks()
    assert [t["task_id"] for t in tasks] == sorted(ids)
