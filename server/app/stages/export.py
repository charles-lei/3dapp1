"""阶段 4：Poisson mesh 导出（ns-export 封装）+ trimesh 清理。"""

from collections.abc import Callable
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

from app.errors import CommandError, PipelineError
from app.interfaces import ProgressCallback
from app.runner import run_command


class PoissonExporter:
    def __init__(self, runner: Callable = run_command):
        self.runner = runner

    def export(self, task_dir: Path, config_path: Path, on_progress: ProgressCallback) -> Path:
        on_progress(10, "渲染深度图")
        out_dir = task_dir / "export"
        try:
            self.runner([
                "ns-export", "poisson",
                "--load-config", str(config_path),
                "--output-dir", str(out_dir),
            ])
        except CommandError as e:
            raise PipelineError(
                "EXPORT_FAILED", f"mesh 导出进程异常退出（exit {e.returncode}），请重试"
            ) from e
        on_progress(70, "Poisson 重建完成")
        ply_files = sorted(out_dir.glob("*.ply"))
        if not ply_files:
            raise PipelineError("EXPORT_FAILED", "导出结束但未找到 .ply 产物")
        return ply_files[0]


def cleanup_mesh(
    mesh_path: Path,
    out_path: Path,
    max_faces: int = 500_000,
    keep_components: int = 3,
) -> dict:
    mesh = trimesh.load(mesh_path, force="mesh")
    faces_in = len(mesh.faces)

    # 1) 连通域清理：保留面积前 keep_components 的簇
    components = mesh.split(only_watertight=False)
    components_removed = max(0, len(components) - keep_components)
    if len(components) > keep_components:
        components = sorted(components, key=lambda c: c.area, reverse=True)[:keep_components]
        mesh = trimesh.util.concatenate(components)

    # 2) 面数抽减（fast quadric simplification）
    if len(mesh.faces) > max_faces:
        vertices, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int64),
            target_reduction=1.0 - max_faces / len(mesh.faces),
        )
        simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # 保留顶点色（简化后按最近顶点取色）
        if mesh.visual.kind == "vertex":
            tree = trimesh.proximity.ProximityQuery(mesh)
            closest = tree.vertex(vertices)[1]
            simplified.visual = trimesh.visual.ColorVisuals(
                simplified, vertex_colors=mesh.visual.vertex_colors[closest]
            )
        mesh = simplified

    # 3) 居中 + 缩放到单位包围盒
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)
    max_extent = float(max(mesh.extents))
    if max_extent <= 0:
        raise PipelineError("EXPORT_FAILED", "网格退化，无有效几何")
    scale = 1.0 / max_extent
    mesh.apply_scale(scale)

    mesh.export(out_path)
    return {
        "faces_in": faces_in,
        "faces_out": len(mesh.faces),
        "components_removed": components_removed,
    }
