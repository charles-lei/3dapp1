import pytest

from app.errors import CommandError, PipelineError
from app.stages.train import SplatfactoTrainer, find_config, is_oom, parse_iteration


def test_parse_iteration():
    assert parse_iteration("5000/10000:  50%|█████", 10000) == 5000
    assert parse_iteration("Printing profiling stats", 10000) is None
    assert parse_iteration("9999/20000", 10000) is None  # max 不匹配


def test_is_oom():
    assert is_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.0 GiB")
    assert not is_oom("everything is fine")


def test_find_config_returns_latest(tmp_path):
    old = tmp_path / "splatfacto" / "2026-07-24_100000" / "config.yml"
    new = tmp_path / "splatfacto" / "2026-07-24_120000" / "config.yml"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("old")
    new.write_text("new")
    assert find_config(tmp_path) == new


def test_trainer_reports_progress(tmp_path):
    task_dir = tmp_path / "t1"
    (task_dir / "colmap").mkdir(parents=True)

    def runner(cmd, on_line=None):
        for it in (2500, 5000, 7500, 10000):
            on_line(f"{it}/10000")
        cfg = task_dir / "train_output" / "splatfacto" / "run1" / "config.yml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("cfg")
        return "done"

    trainer = SplatfactoTrainer(max_iterations=10000, runner=runner)
    events = []
    config_path = trainer.train(task_dir, lambda pct, msg: events.append((pct, msg)))
    assert [p for p, _ in events] == [25, 50, 75, 100]
    assert config_path.name == "config.yml"


def test_trainer_raises_oom(tmp_path):
    task_dir = tmp_path / "t2"
    (task_dir / "colmap").mkdir(parents=True)

    def runner(cmd, on_line=None):
        raise CommandError(cmd, 1, "CUDA out of memory")

    trainer = SplatfactoTrainer(runner=runner)
    with pytest.raises(PipelineError) as exc_info:
        trainer.train(task_dir, lambda p, m: None)
    assert exc_info.value.code == "TRAIN_OOM"


def test_trainer_raises_train_failed_on_other_errors(tmp_path):
    task_dir = tmp_path / "t3"
    (task_dir / "colmap").mkdir(parents=True)

    def runner(cmd, on_line=None):
        raise CommandError(cmd, 2, "segmentation fault")

    trainer = SplatfactoTrainer(runner=runner)
    with pytest.raises(PipelineError) as exc_info:
        trainer.train(task_dir, lambda p, m: None)
    assert exc_info.value.code == "TRAIN_FAILED"
