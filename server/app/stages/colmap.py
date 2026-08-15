"""阶段 2：COLMAP 位姿估计（经 ns-process-data 封装）。"""

import json
from collections.abc import Callable
from pathlib import Path

from app.errors import PipelineError
from app.runner import run_command
from app.status import StatusStore, TaskState

MIN_REGISTRATION_RATIO = 0.7


def registration_ratio(transforms_path: Path, num_input_images: int) -> float:
    with open(transforms_path, encoding="utf-8") as f:
        frames = json.load(f).get("frames", [])
    return len(frames) / max(num_input_images, 1)


def run(
    task_dir: Path,
    store: StatusStore,
    images_subdir: str = "images_clean",
    runner: Callable | None = None,
) -> None:
    runner = runner or run_command
    store.update(state=TaskState.COLMAP, progress=12, stage_detail="COLMAP 特征提取与匹配")
    images_dir = task_dir / images_subdir
    out_dir = task_dir / "colmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner([
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(out_dir),
        # 照片集规模较小（数十至数百张），穷尽匹配更稳；且 COLMAP ≥3.12 已
        # 无法读取 nerfstudio 默认下载的旧版 flann 词袋树（faiss 迁移）。
        "--matching-method", "exhaustive",
    ])
    num_input = len([p for p in images_dir.iterdir() if p.is_file()])
    ratio = registration_ratio(out_dir / "transforms.json", num_input)
    if ratio < MIN_REGISTRATION_RATIO:
        raise PipelineError(
            "COLMAP_LOW_REGISTRATION",
            f"仅 {ratio:.0%} 的图片完成位姿配准（需 ≥ {MIN_REGISTRATION_RATIO:.0%}），"
            "拍摄覆盖不足或物体表面特征太少，请放慢速度、增加环绕角度后重拍",
        )
    store.update(progress=20, stage_detail=f"位姿配准率 {ratio:.0%}")
