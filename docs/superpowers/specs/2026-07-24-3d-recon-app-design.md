# 3D 重建 App MVP 设计文档

- 日期：2026-07-24
- 状态：已评审
- 目标：用户用 Android 手机环绕拍摄物品，系统自动完成 3DGS 重建并导出 GLB，App 内 360° 预览与导出

## 1. 关键决策（已确认）

| 决策点 | 结论 |
|---|---|
| 训练算力 | 本地 NVIDIA GPU 工作站，与手机同局域网，HTTP API 通信 |
| 重建算法 | 3D Gaussian Splatting |
| 管线实现 | Nerfstudio 全家桶（`ns-process-data` + `ns-train splatfacto` + `ns-export poisson`），mesh 提取层预留可替换接口（后续可平移 SuGaR） |
| App 预览 | 训练完成后下载 GLB，本地 SceneView/Filament 渲染，360° 交互 |
| 拍摄引导 | ARCore 轨迹球覆盖提示（半球目标点阵，已覆盖点变色） |
| 采集方式 | 轨迹点触发自动拍照（60~100 张）；视频导入抽帧不在 MVP 范围 |
| 用户模型 | 单用户、任务串行、无鉴权（局域网） |

## 2. 系统架构

```
┌─────────────────────┐         LAN / HTTP          ┌──────────────────────────┐
│   Android App        │ ──────────────────────────▶ │  本地 GPU 工作站           │
│  (Kotlin + Compose)  │                             │  FastAPI 服务 (Python)     │
│                      │ ◀────────────────────────── │                            │
│  · ARCore 轨迹球引导  │   任务状态轮询 / GLB 下载     │  任务队列(单GPU串行)        │
│  · 轨迹点自动拍照      │                             │  管线编排器:               │
│  · 照片打包上传        │                             │   ns-process-data(COLMAP)  │
│  · GLB 360° 预览      │                             │   ns-train splatfacto      │
│                      │                             │   ns-export poisson → GLB  │
└─────────────────────┘                             └──────────────────────────┘
```

核心原则：

- **App 哑客户端**：App 只负责采集、上传、下载、预览；所有重计算在服务端
- **任务异步化**：上传 → 创建任务 → 返回 task_id → App 每 2s 轮询 → 完成后下载 GLB。不用 WebSocket
- **单用户串行**：单 GPU 任务排队依次执行；不引入 Redis/Celery；任务状态存磁盘 JSON（重启可恢复）
- **目录即存储**：每个任务一个目录 `data/tasks/{task_id}/`（`photos/`、`colmap/`、`output/`、`model.glb`、`status.json`），无数据库

## 3. 服务端重建管线

管线编排器（`pipeline.py`）顺序执行四个阶段，每阶段为可独立重试的函数，状态实时写入 `status.json`：

```
uploaded ──▶ preprocessing ──▶ colmap ──▶ training ──▶ mesh_export ──▶ done
```

### 3.1 图像清洗（preprocessing）

- 校验：图片数 ≥ 40、分辨率统一、EXIF 可读；不满足直接失败并返回明确原因
- 模糊检测：拉普拉斯方差，剔除低于阈值（默认 100，可调）的帧
- 重复帧检测：感知哈希去重（防御性保留）
- 清洗后剩余 < 30 张 → 任务失败，提示重拍

### 3.2 COLMAP 位姿估计

- 调 `ns-process-data images --data photos/ --output-dir colmap/`（内部走 COLMAP + 生成 `transforms.json`）
- 失败判据：配准图片比例 < 70% → 失败，提示"拍摄覆盖不足/特征太少，请重拍"

### 3.3 3DGS 训练

- `ns-train splatfacto --data colmap/`，固定 7k~10k 迭代（RTX 4090 约 3~6 分钟）
- 进度上报：解析训练日志 iteration 行，更新 `status.json` 的 `progress`（训练阶段映射到总进度 10%~80%）
- checkpoint 保留在任务目录，便于后续导出 splat .ply

### 3.4 Mesh 导出

- `ns-export poisson --load-config ... --output-dir output/`
- 后处理（自写脚本，trimesh + pymeshlab）：
  - 连通域清理漂浮碎片（保留最大连通簇 + 面积前 N 的簇）
  - 面数抽减 ≤ 50 万面（保证手机端流畅）
  - 居中 + 缩放到单位包围盒
  - 转 GLB；MVP 出**顶点色 mesh**（Poisson 自带颜色），不做 UV 烘焙纹理（后置迭代）

### 3.5 可替换接口

阶段 3、4 抽象为 `Trainer` / `MeshExporter` 协议（Python Protocol）。方案 A 实现为 `SplatfactoTrainer` + `PoissonExporter`；将来换 SuGaR 只加实现类，不动编排器。

### 3.6 失败处理

任一阶段失败 → 状态 `failed` + 结构化错误码 + 中文提示。训练 OOM 自动降分辨率重试一次。

## 4. API 契约

REST + JSON，照片上传走 multipart。无鉴权（局域网），服务端启动时打印本机 IP:Port，App 手动填写或扫码配置。

```
POST   /api/tasks                    创建任务（multipart: photos[]）→ { task_id }
GET    /api/tasks/{id}/status        → { state, progress, stage, error?, eta_seconds? }
GET    /api/tasks/{id}/model         下载 model.glb（支持 Range 断点续传）
GET    /api/tasks/{id}/thumbnail     预览渲染图（任务列表用）
DELETE /api/tasks/{id}               删除任务及数据
```

状态机：`uploaded → preprocessing → colmap → training → mesh_export → done | failed`

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

错误码（failed 时返回，各带 `message_zh` 供 App 直接展示）：

- `INSUFFICIENT_IMAGES` 图片数量不足
- `BLURRY_IMAGES` 模糊图片过多
- `COLMAP_LOW_REGISTRATION` 位姿配准率过低（覆盖不足/特征太少）
- `TRAIN_FAILED` 训练失败
- `EXPORT_FAILED` mesh 导出失败
- `SERVER_BUSY` 服务器忙

## 5. Android 端设计

Kotlin + Jetpack Compose + CameraX + ARCore（SceneView AR 模块）+ SceneView/Filament（GLB 预览）+ Retrofit/OkHttp + Room。MVVM（ViewModel + StateFlow）。

### 5.1 页面

1. **主页/任务列表**：历史任务卡片（缩略图、状态、进度条），轮询活跃任务；"新建扫描"入口；设置服务器地址
2. **拍摄页**（核心）：
   - CameraX 预览 + ARCore session 共享相机
   - 轨迹球引导：以物体为中心生成半球面目标点阵（~72 点：3 个俯仰环 × 24 方位角），渲染为半透明小球；已拍区域点变绿
   - 自动触发：位姿对准未覆盖点（角度差 < 15°）且手机稳定（陀螺仪方差低于阈值持续 0.5s）→ 自动拍照 + 震动/音效反馈
   - 覆盖度进度环；≥ 60 张且覆盖 ≥ 80% 解锁"完成"按钮
   - 首帧引导用户将物体对准中心标记完成"定锚"（物体中心 = ARCore 锚点）
3. **上传/任务详情页**：照片打包 ZIP 上传（进度条）→ 服务端进度 → 完成后自动下载 GLB
4. **预览页**：SceneView/Filament 加载 GLB，单指旋转/双指缩放/平移；导出按钮（分享 GLB / 存公共目录）

### 5.2 内部模块

- `capture/`：ARCore 位姿追踪、轨迹点覆盖判断、自动快门逻辑（纯 Kotlin，可单测）
- `network/`：Retrofit 接口 + multipart 上传（进度回调）+ 轮询器
- `repository/`：TaskRepository（Room 存任务元数据 + 文件路径），杀进程不丢任务

### 5.3 容错

- 上传失败自动重试 3 次
- App 退后台服务端任务继续，回来后靠 task_id 恢复状态
- 服务端不可达时主页顶部横幅提示

## 6. 仓库结构

```
3dapp/
├── server/                    # FastAPI 服务端（Python 3.10, conda env）
│   ├── app/
│   │   ├── main.py            # FastAPI 路由
│   │   ├── tasks.py           # 任务队列（单 worker 线程，串行）
│   │   ├── pipeline.py        # 四阶段编排器
│   │   ├── stages/            # preprocess.py / colmap.py / train.py / export.py
│   │   ├── interfaces.py      # Trainer / MeshExporter 协议
│   │   └── cleanup.py         # 模糊检测、去重、mesh 清理
│   ├── tests/                 # pytest
│   └── data/tasks/            # 任务数据目录
└── android/                   # Gradle 项目
    └── app/src/main/java/com/recon3d/
        ├── ui/                # Compose 页面 + ViewModel
        ├── capture/           # ARCore 轨迹球、自动快门
        ├── network/           # Retrofit、上传、轮询
        └── repository/        # Room + TaskRepository
```

## 7. 测试策略

- **服务端**：pytest 单测覆盖图像清洗阈值逻辑、任务状态机流转、API 契约（mock 管线，不起 GPU）；另备端到端验证脚本——用预拍好、确认可重建的照片集（~80 张）跑完整管线作回归基准，人工检查 GLB 可打开、面数达标
- **Android**：`capture/` 覆盖判断与快门触发逻辑 JUnit 单测（纯数学，不依赖 ARCore）；UI/AR 手测
- **联调验收标准（MVP Done）**：用 App 实拍一个桌面小物件 → 全自动出 GLB → App 内 360° 预览 ≥30fps → GLB 可导入 Blender 正常显示

## 8. 里程碑

1. **M1 服务端管线**：命令行脚本跑通"照片文件夹 → GLB"（不接 API），用现成照片集验证 —— 全项目风险最高，先排雷
2. **M2 API 服务**：FastAPI + 任务队列 + 状态轮询，curl/脚本验证
3. **M3 Android 基础链路**：服务器配置、手动快门拍照、上传、轮询、GLB 预览、分享导出（不含 ARCore 即可端到端用）
4. **M4 ARCore 轨迹球引导**：自动触发拍照 + 覆盖度 UI

## 9. 明确不做（MVP 范围外）

- 视频导入抽帧（MVP 仅轨迹点自动拍照）
- 实时特征点密度反馈
- UV 展开 + 纹理烘焙（MVP 顶点色 mesh）
- 云端部署、多用户、鉴权、任务并发
- 3DGS 远程流式渲染预览、手机端 splat 渲染
- OBJ 导出（GLB 为主；trimesh 转 OBJ 是后置一行代码的事，不阻塞）
