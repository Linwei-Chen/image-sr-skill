#!/usr/bin/env bash
# 一键安装：创建 venv → 安装 torch/pillow/numpy → 下载 Real-ESRGAN x4plus 权重
# 用法: bash setup.sh    （在任意目录运行均可，安装位置固定在本 skill 目录内）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="$ROOT/venv"
WEIGHTS="$ROOT/weights/RealESRGAN_x4plus.pth"
WEIGHTS_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_URL_MIRROR="https://gh-proxy.com/$WEIGHTS_URL"

# 1. 创建 venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "[1/3] 创建虚拟环境 venv ..."
  python3 -m venv "$VENV"
fi

# 2. 安装依赖：python < 3.9 时 torch 最高只到 2.1.2
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
if [ "$PYVER" -lt 39 ]; then TORCH_SPEC="torch==2.1.2"; else TORCH_SPEC="torch"; fi

echo "[2/3] 安装依赖 ($TORCH_SPEC, pillow, numpy) ..."
if ! "$VENV/bin/pip" install -q --timeout 20 --retries 1 "$TORCH_SPEC" pillow numpy; then
  echo "  官方 PyPI 失败/超时，改用清华镜像重试 ..."
  "$VENV/bin/pip" install -q -i https://pypi.tuna.tsinghua.edu.cn/simple "$TORCH_SPEC" pillow numpy
fi

# 3. 下载权重（64MB），官方 GitHub 失败自动换 gh-proxy 镜像
if [ ! -f "$WEIGHTS" ]; then
  echo "[3/3] 下载模型权重 RealESRGAN_x4plus.pth ..."
  mkdir -p "$(dirname "$WEIGHTS")"
  curl -fL --connect-timeout 15 --max-time 600 -o "$WEIGHTS" "$WEIGHTS_URL" \
    || curl -fL --connect-timeout 15 --max-time 600 -o "$WEIGHTS" "$WEIGHTS_URL_MIRROR"
fi

"$VENV/bin/python" - <<'PYEOF'
import torch, PIL, numpy
print("环境就绪: torch", torch.__version__, "| MPS 加速可用:", torch.backends.mps.is_available())
PYEOF
echo "安装完成。超分用法: $VENV/bin/python $ROOT/scripts/sr.py 图片.png"
