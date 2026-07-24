import cv2
import numpy as np
import pytest

from app import pipeline
from app.errors import PipelineError
from app.status import StatusStore
from app.stages import colmap, export


def _make_photos(task_dir, count: int = 3):
    photos = task_dir / "photos"
    photos.mkdir(parents=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(photos / f"{i}.jpg"), img)


class FakeTrainer:
    def __init__(self, fail_with: PipelineError | None = None):
        self.fail_with = fail_with
        self.calls = 0

    def train(self, task_dir, on_progress):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        on_progress(50, "iteration 5000/10000")
        return task_dir / "train_output" / "splatfacto" / "run" / "config.yml"


class FakeExporter:
    def export(self, task_dir, config_path, on_progress):
        import trimesh

        ply = task_dir / "export" / "mesh.ply"
        ply.parent.mkdir(exist_ok=True)
        trimesh.creation.box().export(ply)
        return ply


def _fake_colmap_runner(cmd, on_line=None):
    import json

    out_dir = cmd[cmd.index("--output-dir") + 1]
    with open(f"{out_dir}/transforms.json", "w") as f:
        json.dump({"frames": [{}, {}, {}]}, f)  # 3/3 = 100%
    return "ok"


def test_full_pipeline_succeeds(tmp_path, monkeypatch):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    pipeline.run_pipeline(
        task_dir, trainer=FakeTrainer(), exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "done"
    assert status["progress"] == 100
    assert (task_dir / "model.glb").exists()
    assert (task_dir / "thumbnail.jpg").exists()


def test_pipeline_failure_sets_failed_status(tmp_path, monkeypatch):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    failing = FakeTrainer(fail_with=PipelineError("TRAIN_FAILED", "炸了"))
    pipeline.run_pipeline(
        task_dir, trainer=failing, exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"


def test_oom_triggers_downscale_retry_once(tmp_path, monkeypatch):
    task_dir = tmp_path / "t3"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    trainer = FakeTrainer(fail_with=PipelineError("TRAIN_OOM", "显存不足"))
    pipeline.run_pipeline(
        task_dir, trainer=trainer, exporter=FakeExporter(), min_images=2, min_kept=2
    )
    assert trainer.calls == 2  # 原始 + 降采样重试各一次
    assert (task_dir / "images_lr").exists()  # 降采样图片已生成
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"  # 重试仍失败 → TRAIN_FAILED


def test_unexpected_exception_becomes_train_failed(tmp_path, monkeypatch):
    task_dir = tmp_path / "t4"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)

    class BuggyTrainer:
        def train(self, task_dir, on_progress):
            raise ValueError("bug")

    pipeline.run_pipeline(
        task_dir, trainer=BuggyTrainer(), exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"
    assert "ValueError" in status["error"]["message_zh"]
