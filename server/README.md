# 3D 重建服务端

> 完整文档见仓库根目录 [README](../README.md)（管线说明、API 契约、错误码、里程碑）。

## 环境（GPU 工作站，Ubuntu + NVIDIA）
1. `conda create -n recon python=3.10 -y && conda activate recon`
2. 按 nerfstudio 官方文档安装（含 CUDA 版 PyTorch）：https://docs.nerf.studio/quickstart/installation.html
3. `pip install -r requirements.txt`
4. 验证：`ns-train --help`、`ns-process-data --help`、`ns-export --help` 均可用

## 常用命令
```bash
# M1：命令行跑完整管线（照片目录 → GLB，约 80 张环绕照片）
bash scripts/e2e_m1.sh /path/to/photo_set
# 等价：python -m app.cli --photos /path/to/photo_set

# M2：启动 API 服务（默认 0.0.0.0:8000，启动时打印局域网 IP）
python -m app.main

# 冒烟：先启动服务，再执行
bash scripts/smoke_m2.sh    # 上传 45 张合成图并轮询（预期在 colmap 阶段失败，证明链路通）

# 视频抽帧 + 清洗
python -m app.video_cli --video input.mp4 --out frames/
```

## 单元测试（任何机器，无需 GPU）
```bash
pip install -r requirements.txt && pytest
```
