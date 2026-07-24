# 服务端管线（M1 + M2）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现"照片文件夹 → GLB"的全自动重建管线（M1），并用 FastAPI 包装成局域网 HTTP 服务（M2）。

**Architecture:** Python 包 `app/`：图像清洗 → COLMAP 封装（`ns-process-data`）→ 3DGS 训练封装（`ns-train splatfacto`）→ Poisson mesh 导出 + trimesh 清理。每个阶段是纯函数/类，通过 subprocess 调 nerfstudio CLI，全部可 mock 测试。M2 加 FastAPI 路由 + 单 worker 线程串行任务队列，任务状态落盘 `status.json`。

**Tech Stack:** Python 3.10、FastAPI、uvicorn、opencv-python-headless、imagehash、Pillow、trimesh、fast-simplification、pytest、httpx；nerfstudio 仅安装在 GPU 工作站上（单元测试不需要）。

## Global Constraints

- Python 3.10；服务端代码全部在 `server/` 目录下，包名 `app`
- 无数据库：任务状态唯一来源是 `data/tasks/{task_id}/status.json`
- 错误码集合（见 spec §4）：`INSUFFICIENT_IMAGES`、`BLURRY_IMAGES`、`COLMAP_LOW_REGISTRATION`、`TRAIN_FAILED`、`EXPORT_FAILED`、`SERVER_BUSY`；内部另有 `TRAIN_OOM`（不暴露给 App）
- 总进度映射：`uploaded`=0；`preprocessing` 2→10；`colmap` 12→20；`training` 20→80；`mesh_export` 82→95；`done`=100（对 spec §3.3 "训练 10~80%" 的具体化）
- 所有 nerfstudio 调用必须可注入 runner（subprocess 封装），单元测试不得在 GPU 上跑、不得真调 nerfstudio
- 每张图片校验/清洗下限：上传 ≥ 40 张，清洗后 ≥ 30 张；COLMAP 配准率 ≥ 70%
- mesh 清理：保留面积前 3 的连通簇，面数 ≤ 500,000，居中并缩放到单位包围盒，输出 GLB（顶点色）
- commit 信息用 conventional commits（`feat:`/`test:`/`chore:`）

---

### Task 1: 项目脚手架 + subprocess runner

**Files:**
- Create: `server/requirements.txt`
- Create: `server/pytest.ini`
- Create: `server/app/__init__.py`
- Create: `server/app/errors.py`
- Create: `server/app/runner.py`
- Create: `server/tests/__init__.py`
- Create: `server/tests/test_runner.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces:
  - `app.errors.PipelineError(code: str, message_zh: str)` — `Exception` 子类，属性 `.code`、`.message_zh`
  - `app.errors.CommandError(cmd: list[str], returncode: int, output: str)` — `Exception` 子类
  - `app.runner.run_command(cmd: list[str], on_line: Callable[[str], None] | None = None) -> str` — 运行命令，逐行回调 stdout/stderr 合并输出，返回完整输出文本；非零退出抛 `CommandError`

- [ ] **Step 1: 写脚手架文件**

`server/requirements.txt`:
```
fastapi>=0.110
uvicorn[standard]>=0.29
opencv-python-headless>=4.9
imagehash>=4.3
Pillow>=10.0
trimesh>=4.4
fast-simplification>=0.1.7
pytest>=8.0
httpx>=0.27
python-multipart>=0.0.9
```

`server/pytest.ini`:
```
[pytest]
testpaths = tests
```

`server/app/__init__.py` 和 `server/tests/__init__.py`：空文件。

`.gitignore` 追加：
```
__pycache__/
*.pyc
.venv/
server/data/
```

- [ ] **Step 2: 写 errors.py（无测试依赖的纯定义，先行）**

`server/app/errors.py`:
```python
"""结构化错误定义。code 取值见 spec §4 错误码集合。"""


class PipelineError(Exception):
    """管线业务错误，code 会透传给 App。"""

    def __init__(self, code: str, message_zh: str):
        super().__init__(f"{code}: {message_zh}")
        self.code = code
        self.message_zh = message_zh


class CommandError(Exception):
    """外部命令（COLMAP/nerfstudio）执行失败。"""

    def __init__(self, cmd: list[str], returncode: int, output: str):
        self.cmd = cmd
        self.returncode = returncode
        self.output = output
        super().__init__(f"command failed ({returncode}): {' '.join(cmd)}\n{output[-2000:]}")
```

- [ ] **Step 3: 写失败测试**

`server/tests/test_runner.py`:
```python
import sys

import pytest

from app.errors import CommandError
from app.runner import run_command


def test_run_command_returns_output():
    out = run_command([sys.executable, "-c", "print('hello')"])
    assert "hello" in out


def test_run_command_streams_lines():
    lines = []
    run_command(
        [sys.executable, "-c", "print('a'); print('b')"],
        on_line=lines.append,
    )
    assert lines == ["a", "b"]


def test_run_command_raises_on_nonzero_exit():
    with pytest.raises(CommandError) as exc_info:
        run_command([sys.executable, "-c", "import sys; print('oops'); sys.exit(3)"])
    assert exc_info.value.returncode == 3
    assert "oops" in exc_info.value.output


def test_run_command_captures_stderr():
    out = run_command([sys.executable, "-c", "import sys; sys.stderr.write('err-line\\n')"])
    assert "err-line" in out
```

- [ ] **Step 4: 运行测试确认失败**

Run: `cd server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pytest tests/test_runner.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.runner'`

- [ ] **Step 5: 实现 runner.py**

`server/app/runner.py`:
```python
"""外部命令执行封装。所有 nerfstudio/COLMAP 调用都经由此处，便于测试注入。"""

import subprocess
from collections.abc import Callable

from app.errors import CommandError


def run_command(cmd: list[str], on_line: Callable[[str], None] | None = None) -> str:
    """运行命令，合并 stdout/stderr 逐行回调，返回完整输出。非零退出抛 CommandError。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        output_lines.append(line)
        if on_line is not None:
            on_line(line)
    returncode = proc.wait()
    output = "\n".join(output_lines)
    if returncode != 0:
        raise CommandError(cmd, returncode, output)
    return output
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd server && pytest tests/test_runner.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add server/ .gitignore
git commit -m "feat(server): 脚手架 + subprocess runner 封装"
```

---

### Task 2: 任务状态存储（StatusStore）

**Files:**
- Create: `server/app/status.py`
- Test: `server/tests/test_status.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `app.status.TaskState` — `str` Enum：`UPLOADED/PREPROCESSING/COLMAP/TRAINING/MESH_EXPORT/DONE/FAILED`，值分别为 `"uploaded"/"preprocessing"/"colmap"/"training"/"mesh_export"/"done"/"failed"`
  - `app.status.StatusStore(task_dir: Path)`
    - `.update(state: TaskState | None = None, progress: int | None = None, stage_detail: str | None = None, error: PipelineError | None = None, eta_seconds: int | None = None) -> None` — 原子写入 `task_dir/status.json`；`error` 非 None 时序列化为 `{"code": ..., "message_zh": ...}`
    - `.read() -> dict` — 返回完整 status dict（含 `task_id`，取目录名）；文件不存在抛 `FileNotFoundError`
  - status.json 完整结构：`{"task_id": str, "state": str, "progress": int, "stage_detail": str, "eta_seconds": int | None, "error": {"code": str, "message_zh": str} | None}`

- [ ] **Step 1: 写失败测试**

`server/tests/test_status.py`:
```python
import json

import pytest

from app.errors import PipelineError
from app.status import StatusStore, TaskState


def test_update_then_read_roundtrip(tmp_path):
    task_dir = tmp_path / "abc123"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    status = store.read()
    assert status["task_id"] == "abc123"
    assert status["state"] == "uploaded"
    assert status["progress"] == 0
    assert status["error"] is None


def test_update_merges_partial_fields(tmp_path):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.TRAINING, progress=50, stage_detail="iteration 5000/10000")
    store.update(progress=55)
    status = store.read()
    assert status["state"] == "training"
    assert status["progress"] == 55
    assert status["stage_detail"] == "iteration 5000/10000"


def test_update_with_pipeline_error(tmp_path):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(
        state=TaskState.FAILED,
        error=PipelineError("COLMAP_LOW_REGISTRATION", "位姿配准率过低，请重拍"),
    )
    status = store.read()
    assert status["state"] == "failed"
    assert status["error"] == {
        "code": "COLMAP_LOW_REGISTRATION",
        "message_zh": "位姿配准率过低，请重拍",
    }


def test_read_missing_file_raises(tmp_path):
    store = StatusStore(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        store.read()


def test_update_leaves_no_tmp_files(tmp_path):
    task_dir = tmp_path / "t3"
    task_dir.mkdir()
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    assert [p.name for p in task_dir.iterdir()] == ["status.json"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_status.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.status'`

- [ ] **Step 3: 实现 status.py**

`server/app/status.py`:
```python
"""任务状态存储。status.json 是任务状态的唯一来源（无数据库）。"""

import json
import os
from enum import Enum
from pathlib import Path

from app.errors import PipelineError


class TaskState(str, Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    COLMAP = "colmap"
    TRAINING = "training"
    MESH_EXPORT = "mesh_export"
    DONE = "done"
    FAILED = "failed"


_INITIAL = {
    "state": TaskState.UPLOADED.value,
    "progress": 0,
    "stage_detail": "",
    "eta_seconds": None,
    "error": None,
}


class StatusStore:
    def __init__(self, task_dir: Path):
        self.task_dir = Path(task_dir)
        self.path = self.task_dir / "status.json"

    def read(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            status = json.load(f)
        status["task_id"] = self.task_dir.name
        return status

    def update(
        self,
        state: TaskState | None = None,
        progress: int | None = None,
        stage_detail: str | None = None,
        error: PipelineError | None = None,
        eta_seconds: int | None = None,
    ) -> None:
        status = dict(_INITIAL)
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                status.update(json.load(f))
        if state is not None:
            status["state"] = state.value
        if progress is not None:
            status["progress"] = progress
        if stage_detail is not None:
            status["stage_detail"] = stage_detail
        if eta_seconds is not None:
            status["eta_seconds"] = eta_seconds
        if error is not None:
            status["error"] = {"code": error.code, "message_zh": error.message_zh}
        # 原子写：先写临时文件再 rename，避免读到写了一半的 JSON
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_status.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/status.py server/tests/test_status.py
git commit -m "feat(server): 任务状态存储 StatusStore（原子写 status.json）"
```

---

### Task 3: 图像清洗（模糊检测 + 去重 + 校验）

**Files:**
- Create: `server/app/cleanup.py`
- Test: `server/tests/test_cleanup.py`

**Interfaces:**
- Consumes: `app.errors.PipelineError`
- Produces:
  - `app.cleanup.laplacian_variance(img: np.ndarray) -> float`
  - `app.cleanup.is_blurry(path: Path, threshold: float = 100.0) -> bool`
  - `app.cleanup.phash_str(path: Path) -> str`
  - `app.cleanup.validate_image_set(photo_dir: Path, min_count: int = 40) -> list[Path]` — 返回可读图片列表；数量不足抛 `PipelineError("INSUFFICIENT_IMAGES", ...)`
  - `app.cleanup.CleanResult` — dataclass：`kept: list[Path]`, `dropped_blurry: list[Path]`, `dropped_dupes: list[Path]`
  - `app.cleanup.clean_images(photo_dir: Path, out_dir: Path, blur_threshold: float = 100.0, min_kept: int = 30) -> CleanResult` — 把保留图片复制到 `out_dir`；保留数 < `min_kept` 抛 `PipelineError("BLURRY_IMAGES", ...)`
  - `app.cleanup.downscale_images(src_dir: Path, dst_dir: Path, factor: int = 2) -> list[Path]` — OOM 重试用（Task 6 消费）

- [ ] **Step 1: 写失败测试**

`server/tests/test_cleanup.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_cleanup.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.cleanup'`

- [ ] **Step 3: 实现 cleanup.py**

`server/app/cleanup.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_cleanup.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/cleanup.py server/tests/test_cleanup.py
git commit -m "feat(server): 图像清洗（模糊/去重/校验/降采样）"
```

---

### Task 4: 阶段接口协议 + 预处理阶段

**Files:**
- Create: `server/app/interfaces.py`
- Create: `server/app/stages/__init__.py`（空文件）
- Create: `server/app/stages/preprocess.py`
- Test: `server/tests/test_preprocess.py`

**Interfaces:**
- Consumes: `app.cleanup.*`、`app.status.StatusStore`、`app.errors.PipelineError`
- Produces:
  - `app.interfaces.ProgressCallback = Callable[[int, str], None]` — 参数为（阶段内进度 0~100, 描述文本）
  - `app.interfaces.Trainer` Protocol：`train(self, task_dir: Path, on_progress: ProgressCallback) -> Path`（返回训练产物 `config.yml` 路径）
  - `app.interfaces.MeshExporter` Protocol：`export(self, task_dir: Path, config_path: Path, on_progress: ProgressCallback) -> Path`（返回原始 mesh `.ply` 路径）
  - `app.stages.preprocess.run(task_dir: Path, store: StatusStore) -> None` — 消费 `task_dir/photos/`，产出 `task_dir/images_clean/`；进度 2→10

- [ ] **Step 1: 写接口定义**

`server/app/interfaces.py`:
```python
"""可替换阶段接口。换 SuGaR 等方案时只新增实现类，不动编排器。"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ProgressCallback = Callable[[int, str], None]  # (阶段内进度 0~100, 描述文本)


class Trainer(Protocol):
    def train(self, task_dir: Path, on_progress: ProgressCallback) -> Path:
        """训练并返回产物 config.yml 路径。"""
        ...


class MeshExporter(Protocol):
    def export(self, task_dir: Path, config_path: Path, on_progress: ProgressCallback) -> Path:
        """导出并返回原始 mesh .ply 路径。"""
        ...
```

`server/app/stages/__init__.py`：空文件。

- [ ] **Step 2: 写失败测试**

`server/tests/test_preprocess.py`:
```python
import cv2
import numpy as np
import pytest

from app.errors import PipelineError
from app.stages import preprocess
from app.status import StatusStore, TaskState


def _make_photos(dir_path, count: int, blurry: int = 0):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        if i < blurry:
            img = cv2.GaussianBlur(img, (31, 31), 0)
        cv2.imwrite(str(dir_path / f"{i:03d}.jpg"), img)


def test_run_copies_clean_images_and_updates_status(tmp_path):
    task_dir = tmp_path / "task1"
    _make_photos(task_dir / "photos", count=5)
    store = StatusStore(task_dir)
    store.update(state=TaskState.UPLOADED, progress=0)
    preprocess.run(task_dir, store, min_count=2, min_kept=2)
    assert len(list((task_dir / "images_clean").iterdir())) == 5
    status = store.read()
    assert status["state"] == "preprocessing"
    assert status["progress"] == 10


def test_run_fails_with_insufficient_images(tmp_path):
    task_dir = tmp_path / "task2"
    _make_photos(task_dir / "photos", count=1)
    with pytest.raises(PipelineError) as exc_info:
        preprocess.run(task_dir, StatusStore(task_dir), min_count=40)
    assert exc_info.value.code == "INSUFFICIENT_IMAGES"


def test_run_fails_when_most_images_blurry(tmp_path):
    task_dir = tmp_path / "task3"
    _make_photos(task_dir / "photos", count=5, blurry=5)
    with pytest.raises(PipelineError) as exc_info:
        preprocess.run(task_dir, StatusStore(task_dir), min_count=2, min_kept=30)
    assert exc_info.value.code == "BLURRY_IMAGES"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd server && pytest tests/test_preprocess.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.stages'`

- [ ] **Step 4: 实现 preprocess.py**

`server/app/stages/preprocess.py`:
```python
"""阶段 1：图像清洗。消费 task_dir/photos/，产出 task_dir/images_clean/。"""

from pathlib import Path

from app.cleanup import clean_images, validate_image_set
from app.status import StatusStore, TaskState


def run(task_dir: Path, store: StatusStore, min_count: int = 40, min_kept: int = 30) -> None:
    store.update(state=TaskState.PREPROCESSING, progress=2, stage_detail="校验图片")
    validate_image_set(task_dir / "photos", min_count=min_count)
    store.update(progress=5, stage_detail="模糊检测与去重")
    result = clean_images(task_dir / "photos", task_dir / "images_clean", min_kept=min_kept)
    store.update(
        progress=10,
        stage_detail=f"保留 {len(result.kept)} 张（丢弃模糊 {len(result.dropped_blurry)}、重复 {len(result.dropped_dupes)}）",
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd server && pytest tests/test_preprocess.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add server/app/interfaces.py server/app/stages/ server/tests/test_preprocess.py
git commit -m "feat(server): 阶段接口协议 + 图像清洗阶段"
```

---

### Task 5: COLMAP 阶段（ns-process-data 封装）

**Files:**
- Create: `server/app/stages/colmap.py`
- Test: `server/tests/test_colmap.py`

**Interfaces:**
- Consumes: `app.runner.run_command`、`app.status.StatusStore`、`app.errors.PipelineError`
- Produces:
  - `app.stages.colmap.run(task_dir: Path, store: StatusStore, images_subdir: str = "images_clean", runner: Callable = run_command) -> None`
    - 调 `ns-process-data images --data {task_dir}/{images_subdir} --output-dir {task_dir}/colmap`
    - 解析 `{task_dir}/colmap/transforms.json` 的 `frames` 数，配准率 = frames / 输入图片数 < 0.7 时抛 `PipelineError("COLMAP_LOW_REGISTRATION", ...)`
    - 进度 12→20
  - `app.stages.colmap.registration_ratio(transforms_path: Path, num_input_images: int) -> float`

- [ ] **Step 1: 写失败测试**

`server/tests/test_colmap.py`:
```python
import json

import pytest

from app.errors import PipelineError
from app.stages import colmap
from app.status import StatusStore


def _populate_images(task_dir, count: int, subdir: str = "images_clean"):
    images_dir = task_dir / subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (images_dir / f"{i:03d}.jpg").write_bytes(b"fake")
    return images_dir


def _fake_runner_factory(task_dir, num_frames: int):
    """模拟 ns-process-data：被调时写出 transforms.json。"""

    def fake_runner(cmd, on_line=None):
        out_dir = task_dir / "colmap"
        out_dir.mkdir(exist_ok=True)
        transforms = {"frames": [{"file_path": f"{i}.jpg"} for i in range(num_frames)]}
        (out_dir / "transforms.json").write_text(json.dumps(transforms))
        return "ok"

    return fake_runner


def test_run_invokes_ns_process_data_and_passes(tmp_path):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    _populate_images(task_dir, 10)
    captured = {}

    def runner(cmd, on_line=None):
        captured["cmd"] = cmd
        return _fake_runner_factory(task_dir, 8)(cmd, on_line)

    colmap.run(task_dir, StatusStore(task_dir), runner=runner)
    assert captured["cmd"][0] == "ns-process-data"
    assert "images" in captured["cmd"]
    status = StatusStore(task_dir).read()
    assert status["state"] == "colmap"
    assert status["progress"] == 20


def test_run_fails_on_low_registration(tmp_path):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    _populate_images(task_dir, 10)
    runner = _fake_runner_factory(task_dir, 5)  # 5/10 = 50% < 70%
    with pytest.raises(PipelineError) as exc_info:
        colmap.run(task_dir, StatusStore(task_dir), runner=runner)
    assert exc_info.value.code == "COLMAP_LOW_REGISTRATION"


def test_registration_ratio(tmp_path):
    transforms = tmp_path / "transforms.json"
    transforms.write_text(json.dumps({"frames": [{}, {}, {}]}))
    assert colmap.registration_ratio(transforms, 4) == 0.75
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_colmap.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.stages.colmap'`

- [ ] **Step 3: 实现 colmap.py**

`server/app/stages/colmap.py`:
```python
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
    runner: Callable = run_command,
) -> None:
    store.update(state=TaskState.COLMAP, progress=12, stage_detail="COLMAP 特征提取与匹配")
    images_dir = task_dir / images_subdir
    out_dir = task_dir / "colmap"
    runner([
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(out_dir),
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_colmap.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/stages/colmap.py server/tests/test_colmap.py
git commit -m "feat(server): COLMAP 阶段封装 + 配准率校验"
```

---

### Task 6: 训练阶段（SplatfactoTrainer）

**Files:**
- Create: `server/app/stages/train.py`
- Test: `server/tests/test_train.py`

**Interfaces:**
- Consumes: `app.runner.run_command`、`app.interfaces.ProgressCallback`、`app.errors.PipelineError`
- Produces:
  - `app.stages.train.parse_iteration(line: str, max_iterations: int) -> int | None` — 从日志行解析当前 iteration
  - `app.stages.train.is_oom(output: str) -> bool` — 检测 "CUDA out of memory"
  - `app.stages.train.find_config(train_output_dir: Path) -> Path` — 取最新 `splatfacto/*/config.yml`
  - `app.stages.train.SplatfactoTrainer(max_iterations: int = 10000, runner: Callable = run_command)`，实现 `Trainer` 协议：`train(task_dir, on_progress) -> Path`
    - 命令：`ns-train splatfacto --data {task_dir}/colmap --output-dir {task_dir}/train_output --max-num-iterations {max_iterations} --vis tensorboard`（`--vis tensorboard` 避免拉起 viewer websocket；若安装的 nerfstudio 版本参数名不同，在 M1 真机排雷时按 `ns-train --help` 修正）
    - 输出含 OOM → 抛 `PipelineError("TRAIN_OOM", ...)`；其他非零退出 → `PipelineError("TRAIN_FAILED", ...)`

- [ ] **Step 1: 写失败测试**

`server/tests/test_train.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_train.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.stages.train'`

- [ ] **Step 3: 实现 train.py**

`server/app/stages/train.py`:
```python
"""阶段 3：3DGS 训练（ns-train splatfacto 封装）。"""

import re
from collections.abc import Callable
from pathlib import Path

from app.errors import CommandError, PipelineError
from app.interfaces import ProgressCallback
from app.runner import run_command


def parse_iteration(line: str, max_iterations: int) -> int | None:
    match = re.search(rf"\b(\d+)\s*/\s*{max_iterations}\b", line)
    return int(match.group(1)) if match else None


def is_oom(output: str) -> bool:
    return "CUDA out of memory" in output or "OutOfMemoryError" in output


def find_config(train_output_dir: Path) -> Path:
    configs = sorted(train_output_dir.glob("splatfacto/*/config.yml"))
    if not configs:
        raise PipelineError("TRAIN_FAILED", "训练结束但未找到 config.yml 产物")
    return configs[-1]


class SplatfactoTrainer:
    def __init__(self, max_iterations: int = 10000, runner: Callable = run_command):
        self.max_iterations = max_iterations
        self.runner = runner

    def train(self, task_dir: Path, on_progress: ProgressCallback) -> Path:
        cmd = [
            "ns-train", "splatfacto",
            "--data", str(task_dir / "colmap"),
            "--output-dir", str(task_dir / "train_output"),
            "--max-num-iterations", str(self.max_iterations),
            "--vis", "tensorboard",
        ]
        last_reported = -1

        def on_line(line: str) -> None:
            nonlocal last_reported
            iteration = parse_iteration(line, self.max_iterations)
            if iteration is not None:
                pct = round(iteration / self.max_iterations * 100)
                if pct != last_reported:
                    last_reported = pct
                    on_progress(pct, f"iteration {iteration}/{self.max_iterations}")

        try:
            self.runner(cmd, on_line=on_line)
        except CommandError as e:
            if is_oom(e.output):
                raise PipelineError(
                    "TRAIN_OOM", "显存不足（CUDA OOM），将自动降分辨率重试"
                ) from e
            raise PipelineError(
                "TRAIN_FAILED", f"训练进程异常退出（exit {e.returncode}），请重试或联系开发者"
            ) from e
        return find_config(task_dir / "train_output")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_train.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/stages/train.py server/tests/test_train.py
git commit -m "feat(server): splatfacto 训练封装（进度解析 + OOM 识别）"
```

---

### Task 7: Mesh 导出与清理（PoissonExporter + cleanup_mesh）

**Files:**
- Create: `server/app/stages/export.py`
- Test: `server/tests/test_export.py`

**Interfaces:**
- Consumes: `app.runner.run_command`、`app.interfaces.ProgressCallback`、`app.errors.PipelineError`
- Produces:
  - `app.stages.export.PoissonExporter(runner: Callable = run_command)`，实现 `MeshExporter` 协议：`export(task_dir, config_path, on_progress) -> Path`
    - 命令：`ns-export poisson --load-config {config_path} --output-dir {task_dir}/export`
    - 返回 `{task_dir}/export/` 下第一个 `*.ply`；找不到抛 `PipelineError("EXPORT_FAILED", ...)`
  - `app.stages.export.cleanup_mesh(mesh_path: Path, out_path: Path, max_faces: int = 500_000, keep_components: int = 3) -> dict`
    - 保留面积前 `keep_components` 的连通簇 → 面数超限时用 fast-simplification 抽减 → 按包围盒居中 + 缩放到单位尺寸 → 导出 GLB
    - 返回统计 dict：`{"faces_in": int, "faces_out": int, "components_removed": int}`

- [ ] **Step 1: 写失败测试**

`server/tests/test_export.py`:
```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_export.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.stages.export'`

- [ ] **Step 3: 实现 export.py**

`server/app/stages/export.py`:
```python
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
    scale = 1.0 / max(mesh.extents)
    mesh.apply_scale(scale)

    mesh.export(out_path)
    return {
        "faces_in": faces_in,
        "faces_out": len(mesh.faces),
        "components_removed": components_removed,
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_export.py -v`
Expected: 5 passed（若 `ProximityQuery` 需要 rtree 依赖导致顶点色分支报错，该分支仅在有点色时触发，测试未覆盖；真机排雷时如报错可 `pip install rtree`）

- [ ] **Step 5: Commit**

```bash
git add server/app/stages/export.py server/tests/test_export.py
git commit -m "feat(server): Poisson 导出封装 + mesh 清理（去碎片/抽减/归一化/GLB）"
```

---

### Task 8: 管线编排器（pipeline.py）

**Files:**
- Create: `server/app/pipeline.py`
- Test: `server/tests/test_pipeline.py`

**Interfaces:**
- Consumes: 全部 stages、`app.interfaces`、`app.status.StatusStore`、`app.cleanup.downscale_images`
- Produces:
  - `app.pipeline.run_pipeline(task_dir: Path, trainer: Trainer | None = None, exporter: MeshExporter | None = None, min_images: int = 40, min_kept: int = 30) -> None`
    - 依次执行 preprocess → colmap → train → export → cleanup_mesh，产出 `task_dir/model.glb` 与 `task_dir/thumbnail.jpg`（MVP 缩略图 = 第一张输入照片副本）
    - 训练进度映射到总进度 20→80；导出阶段 82→95；成功置 `done`/100
    - 捕获 `PipelineError`：`TRAIN_OOM` 时把 `images_clean` 降采样 2× 到 `images_lr`，重跑 colmap + train 一次，再失败置 `failed`；其他错误直接置 `failed` 并写入 error
    - 捕获非 `PipelineError` 异常 → `PipelineError("TRAIN_FAILED", ...)` 兜底置 `failed`（不吞异常类型，message 含原始异常类名）

- [ ] **Step 1: 写失败测试**

`server/tests/test_pipeline.py`:
```python
import cv2
import numpy as np
import pytest

from app import pipeline
from app.errors import PipelineError
from app.status import StatusStore
from app.stages import colmap, export


def _make_photos(task_dir, count: int = 3):
    photos = task_dir / "photos"
    photos.mkdir(parents=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(photos / f"{i}.jpg"), img)


class FakeTrainer:
    def __init__(self, fail_with: PipelineError | None = None):
        self.fail_with = fail_with
        self.calls = 0

    def train(self, task_dir, on_progress):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        on_progress(50, "iteration 5000/10000")
        return task_dir / "train_output" / "splatfacto" / "run" / "config.yml"


class FakeExporter:
    def export(self, task_dir, config_path, on_progress):
        import trimesh

        ply = task_dir / "export" / "mesh.ply"
        ply.parent.mkdir(exist_ok=True)
        trimesh.creation.box().export(ply)
        return ply


def _fake_colmap_runner(cmd, on_line=None):
    import json

    out_dir = cmd[cmd.index("--output-dir") + 1]
    with open(f"{out_dir}/transforms.json", "w") as f:
        json.dump({"frames": [{}, {}, {}]}, f)  # 3/3 = 100%
    return "ok"


def test_full_pipeline_succeeds(tmp_path, monkeypatch):
    task_dir = tmp_path / "t1"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    pipeline.run_pipeline(
        task_dir, trainer=FakeTrainer(), exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "done"
    assert status["progress"] == 100
    assert (task_dir / "model.glb").exists()
    assert (task_dir / "thumbnail.jpg").exists()


def test_pipeline_failure_sets_failed_status(tmp_path, monkeypatch):
    task_dir = tmp_path / "t2"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    failing = FakeTrainer(fail_with=PipelineError("TRAIN_FAILED", "炸了"))
    pipeline.run_pipeline(
        task_dir, trainer=failing, exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"


def test_oom_triggers_downscale_retry_once(tmp_path, monkeypatch):
    task_dir = tmp_path / "t3"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)
    trainer = FakeTrainer(fail_with=PipelineError("TRAIN_OOM", "显存不足"))
    pipeline.run_pipeline(
        task_dir, trainer=trainer, exporter=FakeExporter(), min_images=2, min_kept=2
    )
    assert trainer.calls == 2  # 原始 + 降采样重试各一次
    assert (task_dir / "images_lr").exists()  # 降采样图片已生成
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"  # 重试仍失败 → TRAIN_FAILED


def test_unexpected_exception_becomes_train_failed(tmp_path, monkeypatch):
    task_dir = tmp_path / "t4"
    task_dir.mkdir()
    _make_photos(task_dir)
    monkeypatch.setattr(colmap, "run_command", _fake_colmap_runner)

    class BuggyTrainer:
        def train(self, task_dir, on_progress):
            raise ValueError("bug")

    pipeline.run_pipeline(
        task_dir, trainer=BuggyTrainer(), exporter=FakeExporter(), min_images=2, min_kept=2
    )
    status = StatusStore(task_dir).read()
    assert status["state"] == "failed"
    assert status["error"]["code"] == "TRAIN_FAILED"
    assert "ValueError" in status["error"]["message_zh"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_pipeline.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: 实现 pipeline.py**

`server/app/pipeline.py`:
```python
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
```

注意：`colmap.run` 默认参数 `runner=run_command` 在函数定义时绑定，monkeypatch `colmap.run_command` 不会生效——把 `app/stages/colmap.py` 的签名改为 `runner: Callable | None = None`，函数体首行 `runner = runner or run_command`。对 `train.py`/`export.py` 的构造函数注入无此问题（它们在实例化时注入），但 colmap 是模块级函数，必须改。同步更新 `test_colmap.py` 无需变化（测试已显式传 runner）。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: 全量回归 + Commit**

Run: `cd server && pytest -v`
Expected: 全部 passed

```bash
git add server/app/pipeline.py server/app/stages/colmap.py server/tests/test_pipeline.py
git commit -m "feat(server): 管线编排器（四阶段 + OOM 降采样重试）"
```

---

### Task 9: M1 CLI 入口 + 真机端到端验证脚本

**Files:**
- Create: `server/app/cli.py`
- Create: `server/scripts/e2e_m1.sh`
- Create: `server/README.md`

**Interfaces:**
- Consumes: `app.pipeline.run_pipeline`
- Produces:
  - CLI：`cd server && python -m app.cli <photos_dir> [--task-id ID]` — 把照片复制到 `data/tasks/{id}/photos/`，跑完整管线，退出码 0=done 1=failed，终端打印最终 status
  - e2e 脚本：`server/scripts/e2e_m1.sh <photos_dir>` — 在 GPU 工作站上一键验证 M1

- [ ] **Step 1: 写失败测试**

`server/tests/test_cli.py`:
```python
import cv2
import numpy as np

from app import pipeline
from app.cli import main
from app.status import StatusStore, TaskState


def _make_photos(dir_path, count=3):
    dir_path.mkdir(parents=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"{i}.jpg"), img)


def test_cli_runs_pipeline_and_reports_done(tmp_path, monkeypatch, capsys):
    photos = tmp_path / "input"
    _make_photos(photos)
    data_dir = tmp_path / "data"

    def fake_run(task_dir, **kwargs):
        StatusStore(task_dir).update(state=TaskState.DONE, progress=100)

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    rc = main(["--photos", str(photos), "--data-dir", str(data_dir),
               "--task-id", "cli001", "--min-images", "2"])
    assert rc == 0
    assert (data_dir / "tasks" / "cli001" / "photos" / "0.jpg").exists()
    assert '"state": "done"' in capsys.readouterr().out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_cli.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.cli'`

- [ ] **Step 3: 实现 cli.py、e2e 脚本与 README**

`server/app/cli.py`:
```python
"""M1 命令行入口：python -m app.cli --photos <dir> [--data-dir <dir>] [--task-id <id>]"""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path

from app import pipeline
from app.status import StatusStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D 重建管线 CLI（M1）")
    parser.add_argument("--photos", required=True, help="输入照片目录")
    parser.add_argument("--data-dir", default="data", help="任务数据根目录")
    parser.add_argument("--task-id", default=None, help="任务 ID（默认随机生成）")
    parser.add_argument("--min-images", type=int, default=40)
    args = parser.parse_args(argv)

    task_id = args.task_id or secrets.token_hex(3)
    task_dir = Path(args.data_dir) / "tasks" / task_id
    (task_dir / "photos").mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.photos).iterdir()):
        if p.is_file():
            shutil.copy2(p, task_dir / "photos" / p.name)

    pipeline.run_pipeline(task_dir, min_images=args.min_images)
    status = StatusStore(task_dir).read()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if status["state"] == "done":
        print(f"\nGLB 产物: {task_dir / 'model.glb'}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

`server/scripts/e2e_m1.sh`:
```bash
#!/usr/bin/env bash
# M1 端到端验证（在 GPU 工作站上运行）：
#   bash scripts/e2e_m1.sh /path/to/photo_set
# 前置：conda 环境已装 nerfstudio（ns-train --help 可用）+ pip install -r requirements.txt
set -euo pipefail
PHOTOS_DIR="${1:?用法: e2e_m1.sh <photos_dir>}"
cd "$(dirname "$0")/.."
ns-train --help > /dev/null  # 校验 nerfstudio 可用
python -m app.cli --photos "$PHOTOS_DIR"
echo "==> 检查产物：用 Blender/open3d 打开 data/tasks/*/model.glb，确认几何完整、面数 ≤ 50 万"
```

`server/README.md`:
```markdown
# 3D 重建服务端

## 环境（GPU 工作站，Ubuntu + NVIDIA）
1. `conda create -n recon python=3.10 -y && conda activate recon`
2. 按 nerfstudio 官方文档安装（含 CUDA 版 PyTorch）：https://docs.nerf.studio/quickstart/installation.html
3. `pip install -r requirements.txt`
4. 验证：`ns-train --help`、`ns-process-data --help`、`ns-export --help` 均可用

## 单元测试（任何机器，无需 GPU）
pip install -r requirements.txt && pytest

## M1 端到端验证（GPU 工作站）
bash scripts/e2e_m1.sh /path/to/photo_set   # 约 80 张环绕照片

## M2 启动 API 服务
python -m app.main  # 默认 0.0.0.0:8000，启动时打印局域网 IP
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit，M1 代码完成**

```bash
git add server/app/cli.py server/scripts/ server/tests/test_cli.py server/README.md
git commit -m "feat(server): M1 CLI 入口 + e2e 验证脚本"
```

- [ ] **Step 6: 真机排雷（GPU 工作站上人工执行）**

准备一组 ~80 张环绕拍摄的照片（或公开数据集如 NeRF synthetic 的实拍集），运行 `bash scripts/e2e_m1.sh`。预期问题清单（遇到即修，修完补测试）：
- `ns-train splatfacto` 的 `--vis` 参数名/取值随 nerfstudio 版本变化 → 按 `ns-train --help` 修正 `train.py`
- 训练日志 iteration 行格式不符 `parse_iteration` 正则 → 用真实日志行补测试用例
- `ns-export poisson` 产物文件名不是 `*.ply` 首个文件 → 调整 glob
- COLMAP 未安装（nerfstudio 依赖 colmap）→ `conda install -c conda-forge colmap`

---

### Task 10: 任务队列（单 worker 串行 + 重启恢复）

**Files:**
- Create: `server/app/tasks.py`
- Test: `server/tests/test_tasks.py`

**Interfaces:**
- Consumes: `app.pipeline.run_pipeline`、`app.status.StatusStore`
- Produces:
  - `app.tasks.TaskQueue(data_dir: Path, pipeline_fn: Callable[[Path], None] = run_pipeline)`
    - `.create_task(zip_bytes: bytes) -> str` — 生成 6 位 hex task_id，解压 zip 到 `data/tasks/{id}/photos/`（仅接受 .jpg/.jpeg/.png 条目），初始化 status.json 为 `uploaded`，返回 task_id；zip 无有效图片抛 `PipelineError("INSUFFICIENT_IMAGES", ...)`
    - `.submit(task_id: str) -> None` — 放入队列
    - `.start() -> None` — 启动 daemon worker 线程，串行消费队列（每个任务调 `pipeline_fn(task_dir)`，异常吞掉——pipeline 自身已落盘 failed；pipeline_fn 抛出时也兜底置 failed）
    - `.recover() -> int` — 扫描 data_dir，把 state 为活跃态（非 done/failed）的任务重新 submit，返回恢复数量
    - `.task_dir(task_id: str) -> Path`；`.list_tasks() -> list[dict]`（各任务 status dict，按目录名排序）

- [ ] **Step 1: 写失败测试**

`server/tests/test_tasks.py`:
```python
import io
import json
import time
import zipfile

import pytest

from app.errors import PipelineError
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue


def _zip_bytes(names=("a.jpg", "b.jpg")):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"fake-image-bytes")
    return buf.getvalue()


def test_create_task_extracts_zip_and_inits_status(tmp_path):
    q = TaskQueue(tmp_path)
    task_id = q.create_task(_zip_bytes())
    assert len(task_id) == 6
    task_dir = q.task_dir(task_id)
    assert sorted(p.name for p in (task_dir / "photos").iterdir()) == ["a.jpg", "b.jpg"]
    status = StatusStore(task_dir).read()
    assert status["state"] == "uploaded"


def test_create_task_rejects_non_image_entries(tmp_path):
    q = TaskQueue(tmp_path)
    with pytest.raises(PipelineError):
        q.create_task(_zip_bytes(names=("evil.exe",)))


def test_worker_executes_tasks_serially(tmp_path):
    events = []

    def fake_pipeline(task_dir):
        events.append(("start", task_dir.name))
        time.sleep(0.05)
        events.append(("end", task_dir.name))

    q = TaskQueue(tmp_path, pipeline_fn=fake_pipeline)
    ids = [q.create_task(_zip_bytes()) for _ in range(2)]
    for i in ids:
        q.submit(i)
    q.start()
    deadline = time.time() + 5
    while len(events) < 4 and time.time() < deadline:
        time.sleep(0.01)
    # 串行：第一个任务 end 之后才有第二个任务的 start
    assert events == [
        ("start", ids[0]), ("end", ids[0]),
        ("start", ids[1]), ("end", ids[1]),
    ]


def test_recover_resubmits_interrupted_tasks(tmp_path):
    q = TaskQueue(tmp_path, pipeline_fn=lambda d: None)
    done_id = q.create_task(_zip_bytes())
    StatusStore(q.task_dir(done_id)).update(state=TaskState.DONE, progress=100)
    stuck_id = q.create_task(_zip_bytes())
    StatusStore(q.task_dir(stuck_id)).update(state=TaskState.TRAINING, progress=50)
    assert q.recover() == 1  # 只有 training 的被恢复


def test_list_tasks(tmp_path):
    q = TaskQueue(tmp_path)
    ids = [q.create_task(_zip_bytes()) for _ in range(2)]
    tasks = q.list_tasks()
    assert [t["task_id"] for t in tasks] == sorted(ids)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_tasks.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.tasks'`

- [ ] **Step 3: 实现 tasks.py**

`server/app/tasks.py`:
```python
"""任务队列：单 worker 线程串行执行，状态落盘，重启可恢复。"""

import io
import queue
import secrets
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path

from app.cleanup import IMAGE_SUFFIXES
from app.errors import PipelineError
from app.pipeline import run_pipeline
from app.status import StatusStore, TaskState

_ACTIVE_STATES = {
    TaskState.UPLOADED.value,
    TaskState.PREPROCESSING.value,
    TaskState.COLMAP.value,
    TaskState.TRAINING.value,
    TaskState.MESH_EXPORT.value,
}


class TaskQueue:
    def __init__(self, data_dir: Path, pipeline_fn: Callable[[Path], None] = run_pipeline):
        self.data_dir = Path(data_dir)
        self.tasks_root = self.data_dir / "tasks"
        self.pipeline_fn = pipeline_fn
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_root / task_id

    def create_task(self, zip_bytes: bytes) -> str:
        task_id = secrets.token_hex(3)
        photos_dir = self.task_dir(task_id) / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                name = Path(info.filename).name  # 拍平路径，防 zip-slip
                if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                with zf.open(info) as src, open(photos_dir / name, "wb") as dst:
                    dst.write(src.read())
                count += 1
        if count == 0:
            raise PipelineError("INSUFFICIENT_IMAGES", "上传的压缩包中没有可识别的图片")
        StatusStore(self.task_dir(task_id)).update(state=TaskState.UPLOADED, progress=0)
        return task_id

    def submit(self, task_id: str) -> None:
        self._queue.put(task_id)

    def start(self) -> None:
        def work() -> None:
            while True:
                task_id = self._queue.get()
                try:
                    self.pipeline_fn(self.task_dir(task_id))
                except Exception as e:  # pipeline_fn 正常应自行落盘 failed，这里是兜底
                    StatusStore(self.task_dir(task_id)).update(
                        state=TaskState.FAILED,
                        error=PipelineError("TRAIN_FAILED", f"内部错误（{type(e).__name__}）"),
                    )
                finally:
                    self._queue.task_done()

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def recover(self) -> int:
        recovered = 0
        if not self.tasks_root.exists():
            return 0
        for task_dir in sorted(self.tasks_root.iterdir()):
            try:
                status = StatusStore(task_dir).read()
            except FileNotFoundError:
                continue
            if status["state"] in _ACTIVE_STATES:
                self.submit(task_dir.name)
                recovered += 1
        return recovered

    def list_tasks(self) -> list[dict]:
        result = []
        if not self.tasks_root.exists():
            return result
        for task_dir in sorted(self.tasks_root.iterdir()):
            try:
                result.append(StatusStore(task_dir).read())
            except FileNotFoundError:
                continue
        return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_tasks.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/tasks.py server/tests/test_tasks.py
git commit -m "feat(server): 单 worker 串行任务队列 + 重启恢复"
```

---

### Task 11: FastAPI 服务（5 个端点）

**Files:**
- Create: `server/app/main.py`
- Test: `server/tests/test_api.py`

**Interfaces:**
- Consumes: `app.tasks.TaskQueue`、`app.status.StatusStore`
- Produces:
  - `app.main.create_app(data_dir: Path, queue: TaskQueue | None = None) -> FastAPI` — 工厂函数，便于测试注入
  - 端点（契约见 spec §4）：
    - `POST /api/tasks`（multipart 字段名 `file`，内容为 photos zip）→ `201 {"task_id": str}`；空包 400 带 PipelineError JSON
    - `GET /api/tasks/{id}/status` → 200 status dict；不存在 404
    - `GET /api/tasks/{id}/model` → 200 GLB 文件（`model/gltf-binary`）；未完成 409，不存在 404
    - `GET /api/tasks/{id}/thumbnail` → 200 JPEG；无缩略图 404
    - `DELETE /api/tasks/{id}` → 204；任务进行中（非 done/failed）409；不存在 404
  - 模块级 `app.main.app = create_app(...)`（uvicorn 入口）；`python -m app.main` 直接启动并打印局域网 IP

- [ ] **Step 1: 写失败测试**

`server/tests/test_api.py`:
```python
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue


def _zip_bytes(names=("a.jpg",)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"fake-image")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    q = TaskQueue(tmp_path, pipeline_fn=lambda d: None)
    app = create_app(tmp_path, queue=q)
    return TestClient(app), q


def test_create_task(client):
    c, q = client
    resp = c.post("/api/tasks", files={"file": ("photos.zip", _zip_bytes(), "application/zip")})
    assert resp.status_code == 201
    task_id = resp.json()["task_id"]
    status = c.get(f"/api/tasks/{task_id}/status").json()
    assert status["state"] == "uploaded"


def test_create_task_rejects_empty_zip(client):
    c, _ = client
    resp = c.post("/api/tasks", files={"file": ("x.zip", _zip_bytes(names=("a.txt",)), "application/zip")})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INSUFFICIENT_IMAGES"


def test_status_404(client):
    c, _ = client
    assert c.get("/api/tasks/zzzzzz/status").status_code == 404


def test_model_download_requires_done(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.get(f"/api/tasks/{task_id}/model").status_code == 409
    task_dir = q.task_dir(task_id)
    StatusStore(task_dir).update(state=TaskState.DONE, progress=100)
    (task_dir / "model.glb").write_bytes(b"glb-bytes")
    resp = c.get(f"/api/tasks/{task_id}/model")
    assert resp.status_code == 200
    assert resp.content == b"glb-bytes"
    assert resp.headers["content-type"] == "model/gltf-binary"


def test_thumbnail(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.get(f"/api/tasks/{task_id}/thumbnail").status_code == 404
    (q.task_dir(task_id) / "thumbnail.jpg").write_bytes(b"jpeg")
    assert c.get(f"/api/tasks/{task_id}/thumbnail").status_code == 200


def test_delete_task(client):
    c, q = client
    task_id = q.create_task(_zip_bytes())
    assert c.delete(f"/api/tasks/{task_id}").status_code == 409  # uploaded 属活跃态
    StatusStore(q.task_dir(task_id)).update(state=TaskState.DONE, progress=100)
    assert c.delete(f"/api/tasks/{task_id}").status_code == 204
    assert c.get(f"/api/tasks/{task_id}/status").status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd server && pytest tests/test_api.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: 实现 main.py**

`server/app/main.py`:
```python
"""FastAPI 服务入口。启动：python -m app.main（默认 0.0.0.0:8000）。"""

import shutil
import socket
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.errors import PipelineError
from app.status import StatusStore, TaskState
from app.tasks import TaskQueue

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_TERMINAL_STATES = {TaskState.DONE.value, TaskState.FAILED.value}


def create_app(data_dir: Path = DEFAULT_DATA_DIR, queue: TaskQueue | None = None) -> FastAPI:
    task_queue = queue or TaskQueue(data_dir)
    app = FastAPI(title="3D Recon Server")
    app.state.queue = task_queue

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request, exc: PipelineError):
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message_zh": exc.message_zh}},
        )

    @app.post("/api/tasks", status_code=201)
    async def create_task(file: UploadFile):
        task_id = task_queue.create_task(await file.read())
        task_queue.submit(task_id)
        return {"task_id": task_id}

    @app.get("/api/tasks/{task_id}/status")
    async def get_status(task_id: str):
        try:
            return StatusStore(task_queue.task_dir(task_id)).read()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="task not found")

    @app.get("/api/tasks/{task_id}/model")
    async def get_model(task_id: str):
        status = _read_status_or_404(task_queue, task_id)
        if status["state"] != TaskState.DONE.value:
            raise HTTPException(status_code=409, detail="model not ready")
        return FileResponse(
            task_queue.task_dir(task_id) / "model.glb",
            media_type="model/gltf-binary",
            filename="model.glb",
        )

    @app.get("/api/tasks/{task_id}/thumbnail")
    async def get_thumbnail(task_id: str):
        path = task_queue.task_dir(task_id) / "thumbnail.jpg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="thumbnail not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.delete("/api/tasks/{task_id}", status_code=204)
    async def delete_task(task_id: str):
        status = _read_status_or_404(task_queue, task_id)
        if status["state"] not in _TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="task still running")
        shutil.rmtree(task_queue.task_dir(task_id))

    @app.on_event("startup")
    async def startup():
        recovered = task_queue.recover()
        task_queue.start()
        if recovered:
            print(f"恢复 {recovered} 个中断任务")

    return app


def _read_status_or_404(task_queue: TaskQueue, task_id: str) -> dict:
    try:
        return StatusStore(task_queue.task_dir(task_id)).read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="task not found")


def _lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


app = create_app()

if __name__ == "__main__":
    print(f"局域网地址: http://{_lan_ip()}:8000  （在 App 设置中填写此地址）")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd server && pytest tests/test_api.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add server/app/main.py server/tests/test_api.py
git commit -m "feat(server): FastAPI 五个端点 + 启动恢复"
```

---

### Task 12: M2 冒烟验证 + 全量回归

**Files:**
- Create: `server/scripts/smoke_m2.sh`
- Modify: `server/README.md`

**Interfaces:**
- Consumes: 运行中的 API 服务
- Produces: `server/scripts/smoke_m2.sh` — curl 冒烟脚本（非自动化测试，真机/开发机人工跑）

- [ ] **Step 1: 全量回归**

Run: `cd server && pytest -v`
Expected: 全部 passed（约 40 个测试）

- [ ] **Step 2: 写冒烟脚本**

`server/scripts/smoke_m2.sh`:
```bash
#!/usr/bin/env bash
# M2 冒烟：先启动服务（python -m app.main），再在本机执行本脚本。
# 用法: bash scripts/smoke_m2.sh [host:port]
set -euo pipefail
BASE="http://${1:-127.0.0.1:8000}"
TMP="$(mktemp -d)"
mkdir -p "$TMP/photos"
python - "$TMP" <<'PY'
import sys, cv2, numpy as np, pathlib
d = pathlib.Path(sys.argv[1]) / "photos"
for i in range(45):
    rng = np.random.default_rng(i)
    cv2.imwrite(str(d / f"{i:03d}.jpg"), rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
PY
(cd "$TMP/photos" && zip -q "$TMP/photos.zip" ./*)
echo "==> 上传任务"
TASK_ID=$(curl -sf -F "file=@$TMP/photos.zip" "$BASE/api/tasks" | python -c "import sys,json;print(json.load(sys.stdin)['task_id'])")
echo "task_id=$TASK_ID"
echo "==> 轮询状态（随机噪声图会在 colmap 阶段失败，预期看到 COLMAP_LOW_REGISTRATION 或 TRAIN_FAILED）"
for i in $(seq 1 30); do
  RESP=$(curl -sf "$BASE/api/tasks/$TASK_ID/status")
  echo "$RESP"
  STATE=$(echo "$RESP" | python -c "import sys,json;print(json.load(sys.stdin)['state'])")
  [ "$STATE" = "done" ] || [ "$STATE" = "failed" ] && break
  sleep 2
done
echo "==> 冒烟完成（此脚本验证 API 链路；成功出 GLB 的验证用 scripts/e2e_m1.sh + 真实照片）"
```

README 的 M2 一节追加：
```markdown
## 冒烟验证
python -m app.main &        # 启动服务
bash scripts/smoke_m2.sh    # 上传 45 张测试图并轮询状态（预期在 colmap 阶段失败，证明链路通）
```

- [ ] **Step 3: Commit，M2 完成**

```bash
git add server/scripts/smoke_m2.sh server/README.md
git commit -m "feat(server): M2 冒烟脚本 + 文档"
```

---

## Self-Review 记录

- **Spec 覆盖**：spec §3 四个阶段 → Task 4/5/6/7；§3.5 可替换接口 → Task 4/6/7（Protocol + 注入）；§3.6 失败处理与 OOM 重试 → Task 6/8；§4 五端点与错误码 → Task 11；目录即存储/串行队列 → Task 2/10；缩略图 → Task 8（用首张输入照片，spec 允许"MVP 缩略图"未定义渲染方式，此为最简实现）。§3.4 顶点色保留 → Task 7 清理逻辑含点色迁移。视频导入属范围外（spec §9），未实现，正确。
- **已知留白**：nerfstudio CLI 参数名（`--vis`、poisson 产物名）随版本漂移，已在 Task 9 Step 6 列为真机排雷项，并给 `parse_iteration` 等预留了补测试的路径。
- **类型一致性**：`Trainer.train/MeshExporter.export` 签名在 interfaces.py、train.py、export.py、pipeline.py 及全部 Fake 实现中一致；`ProgressCallback` 统一为 `(int, str)`；`StatusStore.update` 参数名各处一致。
