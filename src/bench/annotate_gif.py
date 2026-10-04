"""给已有 GIF 加英文标注（球=Stimulus、挡板=Reflex Paddle、状态行），README 首图用。

  python -m src.bench.annotate_gif
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def annotate(in_path: Path, out_path: Path, status_text: str):
    frames = imageio.mimread(in_path)
    out = []
    for f in frames:
        img = f[:, :, :3].astype(np.int16)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        ball_mask = (r > 180) & (g < 120) & (b < 120)          # 红球
        paddle_mask = (b > 150) & (g > 120) & (r < 120)        # 蓝挡板
        ball = centroid(ball_mask)
        paddle = centroid(paddle_mask)
        im = Image.fromarray(f[:, :, :3]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.text((10, 8), status_text, fill=(255, 255, 255))
        if ball:
            x, y = ball
            d.line([(x, y - 6), (x, y - 40)], fill=(255, 255, 0), width=2)
            d.polygon([(x - 5, y - 40), (x + 5, y - 40), (x, y - 32)], fill=(255, 255, 0))
            d.text((x + 8, y - 48), "Stimulus", fill=(255, 255, 0))
        if paddle:
            x, y = paddle
            d.line([(x, y + 8), (x, y + 36)], fill=(120, 255, 120), width=2)
            d.polygon([(x - 5, y + 36), (x + 5, y + 36), (x, y + 28)], fill=(120, 255, 120))
            d.text((x + 8, y + 30), "Reflex paddle", fill=(120, 255, 120))
        out.append(np.asarray(im))
    imageio.mimsave(out_path, out, duration=0.08)
    print("saved", out_path, len(out), "frames")

if __name__ == "__main__":
    annotate(PROJECT_ROOT / "assets" / "idle_ppo.gif",
             PROJECT_ROOT / "assets" / "idle_ppo_annotated.gif",
             "Vanilla PPO: fidgets when idle (false triggers)")
    annotate(PROJECT_ROOT / "assets" / "idle_dualpath.gif",
             PROJECT_ROOT / "assets" / "idle_dualpath_annotated.gif",
             "RoboReflex dual-pathway: calm cruise, catches on cue")
