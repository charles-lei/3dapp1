import json

import pytest

from app.errors import PipelineError
from app.stages import colmap
from app.status import StatusStore


def _populate_images(task_dir, count: int, subdir: str = "images_clean"):
    images_dir = task_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (images_dir / f"{i:03d}.jpg").write_bytes(b"fake")
    return images_dir


def _fake_runner_factory(task_dir, num_frames: int):
    """模拟 ns-process-data：被调时写出 transforms.json。"""

    def fake_runner(cmd, on_line=None):
        out_dir = task_dir / "colmap"
        out_dir.mkdir(exist_ok=True)
        transforms = {"frames": [{"file_path": f"{i}.jpg"} for i in range(num_frames)]}
        (out_dir / "transforms.json").write_text(json.dumps(transforms))
        return "ok"

    return fake_runner


def test_run_invokes_ns_process_data_and_passes(tmp_path):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    _populate_images(task_dir, 10)
    captured = {}

    def runner(cmd, on_line=None):
        captured["cmd"] = cmd
        return _fake_runner_factory(task_dir, 8)(cmd, on_line)

    colmap.run(task_dir, StatusStore(task_dir), runner=runner)
    assert captured["cmd"][0] == "ns-process-data"
    assert "images" in captured["cmd"]
    status = StatusStore(task_dir).read()
    assert status["state"] == "colmap"
    assert status["progress"] == 20


def test_run_fails_on_low_registration(tmp_path):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    _populate_images(task_dir, 10)
    runner = _fake_runner_factory(task_dir, 5)  # 5/10 = 50% < 70%
    with pytest.raises(PipelineError) as exc_info:
        colmap.run(task_dir, StatusStore(task_dir), runner=runner)
    assert exc_info.value.code == "COLMAP_LOW_REGISTRATION"


def test_registration_ratio(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text(json.dumps({"frames": [{}, {}, {}]}))
    assert colmap.registration_ratio(transforms, 4) == 0.75
