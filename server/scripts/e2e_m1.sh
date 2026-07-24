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
