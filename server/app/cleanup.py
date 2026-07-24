"""图像清洗：可读性校验、模糊检测（拉普拉斯方差）、感知哈希去重、降采样。"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from app.errors import PipelineError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(path: Path, threshold: float = 100.0) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        return True
    return laplacian_variance(img) < threshold


def phash_str(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im))


def _readable_images(photo_dir: Path) -> list[Path]:
    result = []
    for p in sorted(photo_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if cv2.imread(str(p)) is not None:
            result.append(p)
    return result


def validate_image_set(photo_dir: Path, min_count: int = 40) -> list[Path]:
    images = _readable_images(photo_dir)
    if len(images) < min_count:
        raise PipelineError(
            "INSUFFICIENT_IMAGES",
            f"可读图片仅 {len(images)} 张，少于 {min_count} 张下限，请补拍后重试",
        )
    return images


@dataclass
class CleanResult:
    kept: list[Path] = field(default_factory=list)
    dropped_blurry: list[Path] = field(default_factory=list)
    dropped_dupes: list[Path] = field(default_factory=list)


def clean_images(
    photo_dir: Path,
    out_dir: Path,
    blur_threshold: float = 100.0,
    min_kept: int = 30,
) -> CleanResult:
    images = _readable_images(photo_dir)
    result = CleanResult()
    seen_hashes: set[str] = set()
    for path in images:
        h = phash_str(path)
        if h in seen_hashes:
            result.dropped_dupes.append(path)
            continue
        if is_blurry(path, blur_threshold):
            result.dropped_blurry.append(path)
            continue
        seen_hashes.add(h)
        result.kept.append(path)
    if len(result.kept) < min_kept:
        raise PipelineError(
            "BLURRY_IMAGES",
            f"清洗后仅剩 {len(result.kept)} 张清晰图片（需 ≥ {min_kept}），拍摄时请保持稳定并对焦",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in result.kept:
        shutil.copy2(path, out_dir / path.name)
    return result


def downscale_images(src_dir: Path, dst_dir: Path, factor: int = 2) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            resized = im.resize((im.width // factor, im.height // factor), Image.LANCZOS)
            out_path = dst_dir / p.name
            resized.save(out_path)
            out_paths.append(out_path)
    return out_paths
