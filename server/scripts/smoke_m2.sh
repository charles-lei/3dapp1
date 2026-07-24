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
