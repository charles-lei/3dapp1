import numpy as np
import pytest
import trimesh

from app.errors import CommandError, PipelineError
from app.stages.export import PoissonExporter, cleanup_mesh


def test_exporter_invokes_ns_export_and_returns_ply(tmp_path):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("cfg")
    captured = {}

    def runner(cmd, on_line=None):
        captured["cmd"] = cmd
        out_dir = task_dir / "export"
        out_dir.mkdir()
        trimesh.creation.box().export(out_dir / "mesh.ply")
        return "ok"

    exporter = PoissonExporter(runner=runner)
    events = []
    mesh_path = exporter.export(task_dir, config, lambda p, m: events.append(p))
    assert captured["cmd"][:2] == ["ns-export", "poisson"]
    assert str(config) in captured["cmd"]
    assert mesh_path.name == "mesh.ply"


def test_exporter_raises_when_no_ply(tmp_path):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()

    def runner(cmd, on_line=None):
        (task_dir / "export").mkdir()
        return "ok"

    exporter = PoissonExporter(runner=runner)
    with pytest.raises(PipelineError) as exc_info:
        exporter.export(task_dir, tmp_path / "config.yml", lambda p, m: None)
    assert exc_info.value.code == "EXPORT_FAILED"


def test_exporter_wraps_command_error(tmp_path):
    def runner(cmd, on_line=None):
        raise CommandError(cmd, 1, "boom")

    exporter = PoissonExporter(runner=runner)
    with pytest.raises(PipelineError) as exc_info:
        exporter.export(tmp_path, tmp_path / "c.yml", lambda p, m: None)
    assert exc_info.value.code == "EXPORT_FAILED"


def _mesh_with_fragment() -> trimesh.Trimesh:
    big = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    small = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    small.apply_translation((100.0, 100.0, 100.0))  # 远离主体的漂浮碎片
    return trimesh.util.concatenate([big, small])


def test_cleanup_mesh_removes_fragments_and_normalizes(tmp_path):
    src = tmp_path / "mesh.ply"
    _mesh_with_fragment().export(src)
    out = tmp_path / "model.glb"
    stats = cleanup_mesh(src, out, keep_components=1)  # 只留主体，丢弃碎片
    assert stats["components_removed"] == 1
    result = trimesh.load(out, force="mesh")
    # 单位包围盒：最大边长为 1
    assert max(result.extents) == pytest.approx(1.0)
    # 居中：包围盒中心在原点
    assert np.allclose(result.bounds.mean(axis=0), [0, 0, 0], atol=1e-6)


def test_cleanup_mesh_decimates_large_mesh(tmp_path):
    src = tmp_path / "big.ply"
    trimesh.creation.icosphere(subdivisions=5).export(src)  # 20480 面
    out = tmp_path / "model.glb"
    stats = cleanup_mesh(src, out, max_faces=2000)
    # fast quadric 按 target_reduction 逼近目标面数，允许 10% 误差
    assert 0 < stats["faces_out"] <= 2200
    assert stats["faces_in"] == 20480
