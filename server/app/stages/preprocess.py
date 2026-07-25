"""阶段 1：图像清洗。消费 task_dir/photos/，产出 task_dir/images_clean/。"""

from pathlib import Path

import cv2

from app.cleanup import adaptive_blur_threshold, clean_images, laplacian_variance, validate_image_set
from app.status import StatusStore, TaskState


def run(task_dir: Path, store: StatusStore, min_count: int = 40, min_kept: int = 30) -> None:
    store.update(state=TaskState.PREPROCESSING, progress=2, stage_detail="校验图片")
    validated = validate_image_set(task_dir / "photos", min_count=min_count)
    store.update(progress=5, stage_detail="模糊检测与去重")
    variances = [laplacian_variance(cv2.imread(str(p))) for p in validated]
    threshold = adaptive_blur_threshold(variances)
    result = clean_images(task_dir / "photos", task_dir / "images_clean", blur_threshold=threshold, min_kept=min_kept)
    store.update(
        progress=10,
        stage_detail=f"保留 {len(result.kept)} 张（丢弃模糊 {len(result.dropped_blurry)}、重复 {len(result.dropped_dupes)}，阈值 {threshold:.0f}）",
    )
