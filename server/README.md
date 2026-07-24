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

## 冒烟验证
python -m app.main &        # 启动服务
bash scripts/smoke_m2.sh    # 上传 45 张测试图并轮询状态（预期在 colmap 阶段失败，证明链路通）
