import tempfile
from pathlib import Path

import cv2

from app.cleanup import CleanResult, clean_images
from app.errors import PipelineError


def extract_frames(video_path: Path, out_dir: Path, max_frames: int = 120) -> list[Path]:
    if max_frames < 1:
        raise PipelineError("INSUFFICIENT_IMAGES", "max_frames 必须 ≥ 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError("INSUFFICIENT_IMAGES", "视频无法打开或不含有效帧")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise PipelineError("INSUFFICIENT_IMAGES", "视频无法打开或不含有效帧")

        step = max(1, total // max_frames)
        target_indices = set(range(0, total, step))
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        frame_idx = 0
        while len(saved) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_indices:
                dest = out_dir / f"frame_{len(saved) + 1:06d}.jpg"
                cv2.imwrite(str(dest), frame)
                saved.append(dest)
            frame_idx += 1
    finally:
        cap.release()

    return saved


def video_to_photos(
    video_path: Path,
    out_dir: Path,
    max_frames: int = 120,
    blur_threshold: float = 100.0,
    min_kept: int = 30,
) -> CleanResult:
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_dir = Path(temp_dir)
        extract_frames(video_path, raw_dir, max_frames)
        return clean_images(raw_dir, out_dir, blur_threshold, min_kept)
