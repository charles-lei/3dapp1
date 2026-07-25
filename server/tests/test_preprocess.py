import cv2
import numpy as np
import pytest

from app.errors import PipelineError
from app.stages import preprocess
from app.status import StatusStore, TaskState


def _make_photos(dir_path, count: int, blurry: int = 0):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        if i < blurry:
            img = cv2.GaussianBlur(img, (31, 31), 0)
        cv2.imwrite(str(dir_path / f"{i:03d}.jpg"), img)


def test_run_copies_clean_images_and_updates_status(tmp_path):
    task_dir = tmp_path / "task1"
    _make_photos(task_dir / "photos", count=5)
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    preprocess.run(task_dir, store, min_count=2, min_kept=2)
    assert len(list((task_dir / "images_clean").iterdir())) == 5
    status = store.read()
    assert status["state"] == "preprocessing"
    assert status["progress"] == 10


def test_run_fails_with_insufficient_images(tmp_path):
    task_dir = tmp_path / "task2"
    _make_photos(task_dir / "photos", count=1)
    with pytest.raises(PipelineError) as exc_info:
        preprocess.run(task_dir, StatusStore(task_dir), min_count=40)
    assert exc_info.value.code == "INSUFFICIENT_IMAGES"


def test_run_succeeds_with_all_soft_images(tmp_path):
    task_dir = tmp_path / "task4"
    photos = task_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (9, 9), 0)  # variance ~10-60, below the old 100 cap
        cv2.imwrite(str(photos / f"{i:03d}.jpg"), img)

    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    preprocess.run(task_dir, store, min_count=2, min_kept=5)

    status = store.read()
    assert status["state"] == "preprocessing"
    assert status["progress"] == 10
    assert status["stage_detail"].startswith("保留 ")
    assert len(list((task_dir / "images_clean").iterdir())) >= 5


def test_run_fails_when_most_images_blurry(tmp_path):
    task_dir = tmp_path / "task3"
    _make_photos(task_dir / "photos", count=5, blurry=5)
    with pytest.raises(PipelineError) as exc_info:
        preprocess.run(task_dir, StatusStore(task_dir), min_count=2, min_kept=30)
    assert exc_info.value.code == "BLURRY_IMAGES"
