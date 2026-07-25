import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.errors import PipelineError
from app.video import extract_frames, video_to_photos
from app.video_cli import main


def _make_noise_video(path: Path, n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))
    assert writer.isOpened(), f"VideoWriter could not open for {path}"
    for _ in range(n_frames):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_frames_uniform_sampling(tmp_path: Path) -> None:
    video = tmp_path / "noise.mp4"
    out_dir = tmp_path / "frames"
    _make_noise_video(video, 300)

    paths = extract_frames(video, out_dir, max_frames=30)

    assert len(paths) == 30
    assert paths[0].name == "frame_000001.jpg"
    assert all(p.exists() for p in paths)
    assert all(p.parent == out_dir for p in paths)


def test_extract_frames_invalid_video(tmp_path: Path) -> None:
    bad_video = tmp_path / "not_a_video.txt"
    bad_video.write_text("this is not a video")

    with pytest.raises(PipelineError) as exc_info:
        extract_frames(bad_video, tmp_path / "frames")

    assert exc_info.value.code == "INSUFFICIENT_IMAGES"


def _make_blurred_noise_video(path: Path, n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))
    assert writer.isOpened(), f"VideoWriter could not open for {path}"
    for _ in range(n_frames):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # 15x15 blur keeps laplacian variance just above the 10.0 floor but well below 100,
        # so the adaptive threshold is exercised rather than the absolute floor.
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        writer.write(blurred)
    writer.release()


def test_video_to_photos_adaptive_threshold_keeps_soft_frames(tmp_path: Path) -> None:
    video = tmp_path / "soft.mp4"
    out_dir = tmp_path / "photos"
    _make_blurred_noise_video(video, 40)

    result = video_to_photos(video, out_dir, max_frames=40, min_kept=5)

    assert len(result.kept) >= 15


def test_video_to_photos_cleans_and_outputs(tmp_path: Path) -> None:
    video = tmp_path / "noise.mp4"
    out_dir = tmp_path / "photos"
    _make_noise_video(video, 100)

    result = video_to_photos(video, out_dir, max_frames=50, min_kept=5)

    assert result.kept
    assert len(result.kept) <= 50
    for p in result.kept:
        assert (out_dir / p.name).exists()


def test_cli_success_and_bad_input(tmp_path: Path) -> None:
    bad_code = main(["--video", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "out1")])
    assert bad_code == 2

    video = tmp_path / "tiny.mp4"
    out_dir = tmp_path / "out2"
    _make_noise_video(video, 10)

    good_code = main(
        [
            "--video",
            str(video),
            "--out",
            str(out_dir),
            "--min-kept",
            "1",
        ]
    )
    assert good_code == 0
    assert list(out_dir.iterdir())
