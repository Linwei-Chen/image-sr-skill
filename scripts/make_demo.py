#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 README 用的前后对比示例图（合成内容，无版权问题）
模拟真实场景：小尺寸截图/缩略图（原图一半大）→ Real-ESRGAN 4x 放大 → 与原始高清对比
"""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
GW, GH = 600, 450  # 高清参考尺寸


def font(size):
    for p in ("/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc",
              "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_diagram():
    """画一张假的教科书风格示意图（文字+色块+箭头，验证文字与线条重建效果）"""
    img = Image.new("RGB", (GW, GH), "white")
    d = ImageDraw.Draw(img)
    f_big, f_mid, f_small = font(34), font(26), font(20)

    # 细胞膜
    for x in range(10, GW - 20, 20):
        d.ellipse([x, 35, x + 16, 51], outline="#7fb3d5", width=3)
        d.ellipse([x, 63, x + 16, 79], outline="#7fb3d5", width=3)
    d.rectangle([18, 50, GW - 18, 64], fill="#d6eaf8")
    d.text((470, 86), "细胞质", font=f_small, fill="#333")

    # 受体 + 箭头 + 标签
    d.ellipse([70, 12, 145, 72], fill="#f5b7b1", outline="#c0392b", width=3)
    d.text((160, 8), "去甲肾上腺素", font=f_mid, fill="#333")
    d.polygon([(160, 48), (205, 48), (205, 40), (225, 54), (205, 68), (205, 60), (160, 60)], fill="#2ecc71")

    # 步骤文字块
    d.text((45, 125), "ATP", font=f_mid, fill="#1a5276")
    d.text((150, 125), "环化酶", font=f_mid, fill="#333")
    d.text((300, 125), "cAMP + Pi", font=f_mid, fill="#1e8449")
    d.text((120, 168), "2", font=f_mid, fill="#2471a3")
    d.arc([145, 155, 290, 210], 180, 360, fill="#333", width=3)
    d.polygon([(288, 172), (305, 180), (288, 188)], fill="#333")

    d.text((45, 218), "R2C2 PKA", font=f_mid, fill="#333")
    d.text((300, 218), "2C PKA（有活性）", font=f_mid, fill="#1e8449")
    d.text((215, 222), "3", font=f_mid, fill="#2471a3")
    d.polygon([(285, 228), (300, 236), (285, 244)], fill="#333")

    # 线粒体 + 内部结构
    d.ellipse([115, 268, 480, 415], outline="#5dade2", width=5)
    d.ellipse([140, 290, 455, 395], outline="#aed6f1", width=4)
    d.rectangle([195, 318, 390, 366], fill="#abebc6", outline="#145a32", width=3)
    d.text((212, 330), "产热素 质子通道", font=f_small, fill="#145a32")
    d.text((420, 320), "H+", font=f_mid, fill="#c0392b")
    d.text((420, 355), "ATP, ADP", font=f_small, fill="#c0392b")
    d.text((265, 425), "线粒体", font=f_small, fill="#333")
    return img


def main():
    gt = draw_diagram()
    low = gt.resize((GW // 2, GH // 2), Image.LANCZOS)  # 模拟"只有一半大小"的低清输入
    ASSETS.mkdir(exist_ok=True)
    low_path = ASSETS / "demo_low.png"
    low.save(low_path, dpi=(72, 72))

    out_path = ASSETS / "demo_high.png"
    subprocess.run([str(ROOT / "venv" / "bin" / "python"), str(ROOT / "scripts" / "sr.py"),
                    str(low_path), "-o", str(out_path), "--dpi", "72"], check=True)

    # 三联对比（等尺寸展示）：低清输入(最近邻放大) | AI 超分 | 原始高清参考
    panel_low = low.resize((GW, GH), Image.NEAREST)
    panel_sr = Image.open(out_path).convert("RGB").resize((GW, GH), Image.LANCZOS)
    canvas = Image.new("RGB", (GW * 3 + 80, GH + 90), "white")
    canvas.paste(panel_low, (20, 70))
    canvas.paste(panel_sr, (GW + 40, 70))
    canvas.paste(gt, (GW * 2 + 60, 70))
    d = ImageDraw.Draw(canvas)
    f_t = font(24)
    d.text((20, 20), f"输入 {GW//2}×{GH//2}", font=f_t, fill="#c0392b")
    d.text((GW + 40, 20), "image-sr 超分 4×", font=f_t, fill="#1e8449")
    d.text((GW * 2 + 60, 20), "原始高清参考", font=f_t, fill="#2471a3")
    canvas.save(ASSETS / "demo_compare.png")
    print("示例图已生成:", ASSETS / "demo_compare.png")


if __name__ == "__main__":
    main()
