import cv2
import numpy as np

from app import pipeline
from app.cli import main
from app.status import StatusStore, TaskState


def _make_photos(dir_path, count=3):
    dir_path.mkdir(parents=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"{i}.jpg"), img)


def test_cli_runs_pipeline_and_reports_done(tmp_path, monkeypatch, capsys):
    photos = tmp_path / "input"
    _make_photos(photos)
    data_dir = tmp_path / "data"

    def fake_run(task_dir, **kwargs):
        StatusStore(task_dir).update(state=TaskState.DONE, progress=100)

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    rc = main(["--photos", str(photos), "--data-dir", str(data_dir),
               "--task-id", "cli001", "--min-images", "2"])
    assert rc == 0
    assert (data_dir / "tasks" / "cli001" / "photos" / "0.jpg").exists()
    assert '"state": "done"' in capsys.readouterr().out
