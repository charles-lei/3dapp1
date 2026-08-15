# 3D 重建 App（3dapp）

用 Android 手机环绕拍摄物品，局域网内的 GPU 工作站自动完成 **3D Gaussian Splatting 重建**并导出 GLB，App 内 360° 预览与分享。

> 完整设计见 [MVP 设计文档](docs/superpowers/specs/2026-07-24-3d-recon-app-design.md)。

## 系统架构

```
┌─────────────────────┐         LAN / HTTP          ┌──────────────────────────┐
│   Android App        │ ──────────────────────────▶ │  本地 GPU 工作站           │
│  (Kotlin + Compose)  │                             │  FastAPI 服务 (Python)     │
│                      │ ◀─────────────────────────▶ │                            │
│  · ARCore 轨迹球引导  │   任务状态轮询 / GLB 下载     │  任务队列(单GPU串行)        │
│  · 轨迹点自动拍照      │                             │  管线编排器:               │
│  · GLB 360° 预览      │                             │   图像清洗 → COLMAP 位姿   │
│                      │                             │   → splatfacto 3DGS 训练   │
│                      │                             │   → Poisson 导出 → GLB     │
└─────────────────────┘                             └──────────────────────────┘
```

核心原则：

- **App 哑客户端**：采集、上传、下载、预览在 App；所有重计算在服务端
- **任务异步化**：上传 → 创建任务返回 `task_id` → App 每 2s 轮询 → 完成后下载 GLB（不用 WebSocket）
- **单用户串行**：单 GPU 排队执行，不引入 Redis/Celery；状态存磁盘 JSON，重启可恢复
- **目录即存储**：每个任务一个目录 `server/data/tasks/{task_id}/`（`photos/`、`colmap/`、`model.glb`、`status.json`…），无数据库

## 仓库结构

```
3dapp1/
├── server/                     # FastAPI 服务端（Python 3.10, conda 环境）
│   ├── app/
│   │   ├── main.py             # FastAPI 入口与 5 个 API 端点
│   │   ├── tasks.py            # 单 worker 串行任务队列（落盘、重启恢复）
│   │   ├── pipeline.py         # 四阶段编排器（含 OOM 自动降采样重试）
│   │   ├── status.py           # status.json 读写（原子写，状态唯一来源）
│   │   ├── cli.py              # M1 命令行入口：照片目录 → GLB
│   │   ├── video.py            # 视频抽帧 + 自适应模糊清洗
│   │   ├── video_cli.py        # 视频抽帧 CLI 入口
│   │   ├── runner.py           # 外部命令封装（ns-*/COLMAP，逐行回调，便于测试注入）
│   │   ├── errors.py           # 结构化错误（code + 中文提示，透传给 App）
│   │   ├── cleanup.py          # 图像清洗：模糊检测（拉普拉斯方差）/ phash 去重 / 降采样
│   │   ├── interfaces.py       # Trainer / MeshExporter 可替换协议（预留 SuGaR）
│   │   └── stages/
│   │       ├── preprocess.py   # 阶段 1：图像清洗
│   │       ├── colmap.py       # 阶段 2：位姿估计（ns-process-data）
│   │       ├── train.py        # 阶段 3：3DGS 训练（ns-train splatfacto）
│   │       └── export.py       # 阶段 4：Poisson 导出 + trimesh 清理 → GLB
│   ├── scripts/
│   │   ├── e2e_m1.sh           # M1 端到端验证（GPU 工作站）
│   │   └── smoke_m2.sh         # M2 API 冒烟（合成噪声图，验证链路）
│   ├── tests/                  # pytest 单测（mock 管线，无需 GPU）
│   └── data/tasks/             # 任务数据目录（git 忽略）
├── docs/
│   └── superpowers/specs/2026-07-24-3d-recon-app-design.md   # MVP 设计文档
└── android/                    # （规划中）Kotlin + Compose App
```

## 服务端快速开始

### 1. 环境（GPU 工作站，Ubuntu + NVIDIA）

```bash
conda create -n recon python=3.10 -y && conda activate recon
# 按 nerfstudio 官方文档安装（含 CUDA 版 PyTorch）：
#   https://docs.nerf.studio/quickstart/installation.html
pip install -r server/requirements.txt
# 验证
ns-train --help && ns-process-data --help && ns-export --help
```

### 2. M1：命令行跑完整管线（照片目录 → GLB）

```bash
cd server
python -m app.cli --photos /path/to/photo_set      # 约 80 张环绕照片
# 等价于: bash scripts/e2e_m1.sh /path/to/photo_set
# 产物: server/data/tasks/<task_id>/model.glb
```

### 3. M2：启动 API 服务

```bash
python -m app.main          # 默认 0.0.0.0:8000，启动时打印局域网 IP
bash scripts/smoke_m2.sh    # 冒烟：上传 45 张合成图并轮询（预期在 colmap 阶段失败，证明链路通）
```

### 4. 视频抽帧（辅助工具）

```bash
python -m app.video_cli --video input.mp4 --out frames/ \
    [--max-frames 120] [--keep-ratio 0.75] [--min-kept 30]
# 均匀抽帧 + 自适应模糊阈值清洗（阈值 = clamp(分位数, 10, 100)，视频帧不会被全灭）
```

### 5. 单元测试（任何机器，无需 GPU）

```bash
cd server
pip install -r requirements.txt && pytest
```

## 重建管线

```
uploaded → preprocessing → colmap → training → mesh_export → done | failed
```

| 阶段 | 工具 | 说明 | 总进度 |
|---|---|---|---|
| **preprocess** 图像清洗 | OpenCV + imagehash | 校验可读图片 ≥ 40 张；模糊检测（拉普拉斯方差，自适应阈值 `clamp(p25, 10, 100)`）；phash 感知哈希去重；清洗后 < 30 张 → 失败提示重拍 | 0–10 |
| **colmap** 位姿估计 | `ns-process-data images` | 穷尽匹配（小照片集更稳）；配准率 < 70% → 失败提示"覆盖不足/特征太少，请重拍" | 10–20 |
| **training** 3DGS 训练 | `ns-train splatfacto` | 固定 10k 迭代（RTX 4090 约 3–6 分钟）；解析训练日志 iteration 行上报进度；**OOM 时自动降采样 2× 重试一次**，仍 OOM → 失败提示减少图片数量 | 20–80 |
| **mesh_export** mesh 导出 | `ns-export poisson` + trimesh | 连通域清理（保留面积前 3 簇）→ 面数抽减 ≤ 50 万面（fast-simplification，保留顶点色）→ 居中 + 缩放至单位包围盒 → 转 GLB（顶点色，不做 UV 烘焙） | 80–95 |

任一阶段失败 → 状态 `failed` + 结构化错误码 + 中文提示（`message_zh` 可直接展示给 App 用户）。

阶段 3/4 抽象为 `Trainer` / `MeshExporter` 协议（`interfaces.py`）：当前实现为 `SplatfactoTrainer` + `PoissonExporter`，将来换 SuGaR 只加实现类，不动编排器。

## API 契约

REST + JSON，照片上传走 multipart。无鉴权（局域网），服务端启动时打印本机 IP:Port。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/tasks` | 创建任务（multipart 字段 `file` = 照片 ZIP，≤ 500 张 / 5GB）→ `201 { task_id }` |
| `GET` | `/api/tasks/{id}/status` | → `{ task_id, state, progress, stage_detail, eta_seconds, error }` |
| `GET` | `/api/tasks/{id}/model` | 下载 `model.glb`（未完成时 `409`） |
| `GET` | `/api/tasks/{id}/thumbnail` | 缩略图（任务列表用，无则 `404`） |
| `DELETE` | `/api/tasks/{id}` | 删除任务及数据（运行中 `409`） |

status 响应示例：

```json
{
  "task_id": "a3f9c2",
  "state": "training",
  "progress": 55,
  "stage_detail": "iteration 5200/10000",
  "eta_seconds": 180,
  "error": null
}
```

错误码（`failed` 时 `error.code`，均带 `message_zh`）：

| code | 含义 |
|---|---|
| `INSUFFICIENT_IMAGES` | 图片数量不足 / 无有效图片 / 超出上传上限 |
| `BLURRY_IMAGES` | 清洗后清晰图片不足 |
| `COLMAP_LOW_REGISTRATION` | 位姿配准率 < 70%（覆盖不足/特征太少） |
| `TRAIN_OOM` | 显存不足（内部码，触发自动降采样重试，不直接暴露为最终失败） |
| `TRAIN_FAILED` | 训练失败（含降采样重试后仍 OOM、未预期异常兜底） |
| `EXPORT_FAILED` | mesh 导出失败 |

任务存储与队列（`tasks.py` / `status.py`）：

- 每个任务一个目录，`status.json` 为状态唯一来源（临时文件 + rename 原子写）
- 单 worker 线程串行消费；上传即入队
- 服务重启时自动把处于活跃状态的任务重新入队（`recover`）
- 上传包解压前预审：限制图片数量/总大小，拍平路径防 zip-slip

## Android 端（规划中）

Kotlin + Jetpack Compose + CameraX + ARCore + SceneView/Filament，详见设计文档 §5。核心页面：任务列表 / 拍摄页（ARCore 轨迹球引导 + 自动快门）/ 上传与进度 / GLB 360° 预览。

## 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M1** 服务端管线 | 命令行跑通"照片文件夹 → GLB"（`app.cli` + `e2e_m1.sh`） | ✅ 已完成 |
| **M2** API 服务 | FastAPI + 任务队列 + 状态轮询（`app.main` + `smoke_m2.sh`） | ✅ 已完成 |
| **M3** Android 基础链路 | 服务器配置、手动拍照、上传、轮询、GLB 预览、分享导出 | ⏳ 待开始 |
| M4 | ARCore 轨迹球引导：自动触发拍照 + 覆盖度 UI | ⏳ 待开始 |

另有视频抽帧工具（`app.video_cli`）作为照片采集的补充路径（设计文档将视频导入列为 MVP 范围外，但抽帧后走同一管线可复用）。

## 测试

- **单元测试**：`server/tests/`，pytest 覆盖图像清洗阈值逻辑、状态机流转、API 契约、各阶段封装（mock 外部命令，不起 GPU）。`tests/obj1/` 为一组 90 帧抽帧测试数据。
- **端到端回归**：`scripts/e2e_m1.sh` + 预拍好、确认可重建的照片集（~80 张），人工检查 GLB 可打开、面数达标。
- **联调验收（MVP Done）**：App 实拍一个桌面小物件 → 全自动出 GLB → App 内 360° 预览 ≥ 30fps → GLB 可导入 Blender 正常显示。
