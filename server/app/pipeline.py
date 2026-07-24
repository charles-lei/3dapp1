"""管线编排器：preprocess → colmap → train → export → model.glb。

进度映射（Global Constraints）：preprocess 0-10，colmap 10-20，train 20-80，
mesh_export 80-95，done 100。
"""

import shutil
from pathlib import Path

from app.cleanup import downscale_images
from app.errors import PipelineError
from app.interfaces import MeshExporter, Trainer
from app.stages import colmap, export, preprocess
from app.stages.train import SplatfactoTrainer
from app.status import StatusStore, TaskState

TRAIN_PROGRESS_RANGE = (20, 80)
EXPORT_PROGRESS_RANGE = (82, 95)


def _map_progress(on_store_update, lo: int, hi: int):
    def mapped(pct: int, detail: str) -> None:
        on_store_update(progress=lo + round((hi - lo) * pct / 100), stage_detail=detail)

    return mapped


def _set_thumbnail(task_dir: Path) -> None:
    photos = sorted((task_dir / "photos").iterdir())
    if photos:
        shutil.copy2(photos[0], task_dir / "thumbnail.jpg")


def _colmap_and_train(task_dir, store, trainer, images_subdir: str):
    colmap.run(task_dir, store, images_subdir=images_subdir)
    return trainer.train(
        task_dir, _map_progress(store.update, *TRAIN_PROGRESS_RANGE)
    )


def run_pipeline(
    task_dir: Path,
    trainer: Trainer | None = None,
    exporter: MeshExporter | None = None,
    min_images: int = 40,
    min_kept: int = 30,
) -> None:
    store = StatusStore(task_dir)
    trainer = trainer or SplatfactoTrainer()
    exporter = exporter or export.PoissonExporter()
    try:
        preprocess.run(task_dir, store, min_count=min_images, min_kept=min_kept)
        try:
            config_path = _colmap_and_train(task_dir, store, trainer, "images_clean")
        except PipelineError as e:
            if e.code != "TRAIN_OOM":
                raise
            # OOM：降采样 2× 后重跑 colmap + train，仅重试一次
            store.update(stage_detail="显存不足，降分辨率重试中")
            downscale_images(task_dir / "images_clean", task_dir / "images_lr", factor=2)
            try:
                config_path = _colmap_and_train(task_dir, store, trainer, "images_lr")
            except PipelineError as retry_error:
                if retry_error.code == "TRAIN_OOM":
                    raise PipelineError(
                        "TRAIN_FAILED", "降分辨率后仍显存不足，请减少图片数量后重试"
                    ) from retry_error
                raise

        store.update(state=TaskState.MESH_EXPORT, progress=EXPORT_PROGRESS_RANGE[0])
        ply_path = exporter.export(
            task_dir, config_path, _map_progress(store.update, *EXPORT_PROGRESS_RANGE)
        )
        stats = export.cleanup_mesh(ply_path, task_dir / "model.glb")
        _set_thumbnail(task_dir)
        store.update(
            state=TaskState.DONE,
            progress=100,
            stage_detail=f"完成（{stats['faces_out']} 面）",
            eta_seconds=0,
        )
    except PipelineError as e:
        store.update(state=TaskState.FAILED, error=e)
    except Exception as e:  # 兜底：未预期异常统一转 TRAIN_FAILED
        store.update(
            state=TaskState.FAILED,
            error=PipelineError("TRAIN_FAILED", f"内部错误（{type(e).__name__}），请联系开发者"),
        )
