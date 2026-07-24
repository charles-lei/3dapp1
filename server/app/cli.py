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
