import json

import pytest

from app.errors import PipelineError
from app.status import StatusStore, TaskState


def test_update_then_read_roundtrip(tmp_path):
    task_dir = tmp_path / "abc123"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    status = store.read()
    assert status["task_id"] == "abc123"
    assert status["state"] == "uploaded"
    assert status["progress"] == 0
    assert status["error"] is None


def test_update_merges_partial_fields(tmp_path):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.TRAINING, progress=50, stage_detail="iteration 5000/10000")
    store.update(progress=55)
    status = store.read()
    assert status["state"] == "training"
    assert status["progress"] == 55
    assert status["stage_detail"] == "iteration 5000/10000"


def test_update_with_pipeline_error(tmp_path):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(
        state=TaskState.FAILED,
        error=PipelineError("COLMAP_LOW_REGISTRATION", "位姿配准率过低，请重拍"),
    )
    status = store.read()
    assert status["state"] == "failed"
    assert status["error"] == {
        "code": "COLMAP_LOW_REGISTRATION",
        "message_zh": "位姿配准率过低，请重拍",
    }


def test_read_missing_file_raises(tmp_path):
    store = StatusStore(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        store.read()


def test_update_leaves_no_tmp_files(tmp_path):
    task_dir = tmp_path / "t3"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    assert [p.name for p in task_dir.iterdir()] == ["status.json"]
