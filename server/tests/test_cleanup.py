import cv2
import numpy as np
import pytest
from PIL import Image

from app.cleanup import (
    clean_images,
    downscale_images,
    is_blurry,
    laplacian_variance,
    validate_image_set,
)
from app.errors import PipelineError


def _save(img: np.ndarray, path):
    cv2.imwrite(str(path), img)


def _sharp_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


def _blurry_image(seed: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(seed), (31, 31), 0)


def test_sharp_image_has_high_laplacian_variance():
    assert laplacian_variance(_sharp_image()) > 500


def test_blurry_image_has_low_laplacian_variance():
    assert laplacian_variance(_blurry_image()) < 50


def test_is_blurry(tmp_path):
    sharp = tmp_path / "sharp.jpg"
    blurry = tmp_path / "blurry.jpg"
    _save(_sharp_image(), sharp)
    _save(_blurry_image(), blurry)
    assert not is_blurry(sharp)
    assert is_blurry(blurry)


def test_validate_image_set_rejects_too_few(tmp_path):
    for i in range(3):
        _save(_sharp_image(i), tmp_path / f"{i}.jpg")
    with pytest.raises(PipelineError) as exc_info:
        validate_image_set(tmp_path, min_count=40)
    assert exc_info.value.code == "INSUFFICIENT_IMAGES"


def test_validate_image_set_skips_unreadable(tmp_path):
    for i in range(2):
        _save(_sharp_image(i), tmp_path / f"{i}.jpg")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("hello")
    assert len(validate_image_set(tmp_path, min_count=2)) == 2


def test_clean_images_drops_blurry_and_dupes(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    base = _sharp_image(42)
    _save(base, photos / "a.jpg")
    _save(base, photos / "a_copy.jpg")  # 完全相同的副本 → 去重丢弃
    _save(_blurry_image(1), photos / "blur.jpg")
    out = tmp_path / "clean"
    result = clean_images(photos, out, min_kept=1)
    assert [p.name for p in result.kept] == ["a.jpg"]
    assert [p.name for p in result.dropped_dupes] == ["a_copy.jpg"]
    assert [p.name for p in result.dropped_blurry] == ["blur.jpg"]
    # 保留图已复制到 out_dir
    assert (out / "a.jpg").exists()


def test_clean_images_raises_when_too_few_kept(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _save(_blurry_image(1), photos / "blur.jpg")
    with pytest.raises(PipelineError) as exc_info:
        clean_images(photos, tmp_path / "clean", min_kept=30)
    assert exc_info.value.code == "BLURRY_IMAGES"


def test_downscale_images(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _save(_sharp_image(), src / "x.jpg")
    out_paths = downscale_images(src, tmp_path / "dst", factor=2)
    with Image.open(out_paths[0]) as im:
        assert im.size == (320, 240)
