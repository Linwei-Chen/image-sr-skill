#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-ESRGAN x4plus 图像超分辨率（本地推理，无需 GPU 服务）
- 默认 4 倍放大，可选 2 倍；自动分块处理大图，防止内存溢出
- Apple Silicon 自动使用 MPS 加速；输出 PNG（默认 300 DPI）

用法:
  python sr.py 输入.png [-o 输出.png] [--scale 4] [--dpi 300] [--tile 256] [--cpu]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "RealESRGAN_x4plus.pth"


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """与官方 RealESRGAN_x4plus.pth 权重一一对应的网络结构"""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.lrelu(self.conv_hr(feat))
        return self.conv_last(out)


def load_model(weights_path, device):
    model = RRDBNet()
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def sr_tile(model, tile, device):
    """tile: float32 RGB, shape (3, h, w)。返回 4 倍超分结果 (3, 4h, 4w)"""
    t = torch.from_numpy(tile[None]).to(device)
    with torch.no_grad():
        o = model(t)
    return o.squeeze(0).clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Real-ESRGAN x4plus 图像超分")
    ap.add_argument("input", help="输入图片路径")
    ap.add_argument("-o", "--output", help="输出路径（默认与原图同目录，命名 <原名>_x4.png）")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4], help="放大倍数（默认 4）")
    ap.add_argument("--dpi", type=int, default=300, help="输出 PNG 的 DPI（默认 300）")
    ap.add_argument("--tile", type=int, default=256, help="分块大小，内存不足时调小（默认 256）")
    ap.add_argument("--pad", type=int, default=16, help="分块重叠像素，消除接缝（默认 16）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="模型权重路径")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"找不到权重文件 {args.weights}，请先运行 scripts/setup.sh")

    img = Image.open(args.input).convert("RGB")
    W, H = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0  # RGB (H, W, 3)

    if args.cpu or not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "mps"
    print(f"输入 {W}x{H}，设备 {device}", file=sys.stderr)

    model = load_model(args.weights, device)

    SCALE = 4  # 网络固定 4 倍；2 倍由结果缩小得到
    TILE, PAD = args.tile, args.pad
    out = np.zeros((H * SCALE, W * SCALE, 3), dtype=np.float32)

    tiles = [((x, y) ) for y in range(0, H, TILE) for x in range(0, W, TILE)]
    t0 = time.time()
    for i, (x, y) in enumerate(tiles, 1):
        x2, y2 = min(x + TILE, W), min(y + TILE, H)
        xa, xb = max(x - PAD, 0), min(x2 + PAD, W)
        ya, yb = max(y - PAD, 0), min(y2 + PAD, H)
        patch = arr[ya:yb, xa:xb].transpose(2, 0, 1)  # (3, h, w)
        o = sr_tile(model, patch, device)  # (3, 4h, 4w)
        oy, ox = (y - ya) * SCALE, (x - xa) * SCALE
        out[y * SCALE:y2 * SCALE, x * SCALE:x2 * SCALE] = \
            o.transpose(1, 2, 0)[oy:oy + (y2 - y) * SCALE, ox:ox + (x2 - x) * SCALE]
        print(f"  分块 {i}/{len(tiles)}  {time.time() - t0:.0f}s", file=sys.stderr)

    result = Image.fromarray((out * 255).round().clip(0, 255).astype(np.uint8))
    if args.scale == 2:
        result = result.resize((W * 2, H * 2), Image.LANCZOS)

    out_path = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_x4.png"))
    result.save(out_path, dpi=(args.dpi, args.dpi))
    print(f"完成: {out_path}  {result.size[0]}x{result.size[1]}  {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
