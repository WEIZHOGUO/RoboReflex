"""制作 README 首屏英雄 GIF：左 PPO 单通路 vs 右机映双通路 分屏 A/B 对比。

  python -m src.bench.make_hero_gif

产物：assets/hero_compare.gif（1280x480 @25fps，<=5MB，5~8s）
     assets/hero_compare_check.png（6 帧抽帧自检条）

视觉元素：空闲期拉长场景（发球延迟 2.5~3.5s，同 record_idle_compare）、
HUD 实时"刺激->首动"延迟条（双通路反射接管时绿框亮起）、挡板 10 帧渐隐拖尾、
球=Stimulus / 挡板=Reflex paddle 标注（质心追踪同 annotate_gif）、
结尾把双通路接球段（发球前 0.3s -> 接住）以 0.15x 慢放重播。
"""
from __future__ import annotations

import ctypes
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from ..reflex.dualpath_env import DualPathEnv
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_GIF = PROJECT_ROOT / "assets" / "hero_compare.gif"
OUT_STRIP = PROJECT_ROOT / "assets" / "hero_compare_check.png"

FPS = 25                    # 输出帧率（控制 50Hz / STRIDE 2）
STRIDE = 2
PANEL_W, PANEL_H = 640, 480
W, H = 1280, 480
TRAIL_LEN = 10              # 残影拖尾帧数（8~12 区间）
SLOWMO_RATE = 0.15          # 慢放倍速
SLOWMO_LEAD_S = 0.3         # 慢放起点：发球前 0.3s
TIME_BUDGET_S = 7.9         # 总时长预算（规格 5~8s）
SIZE_LIMIT = 5 * 1024 * 1024

LABEL_PPO = "PPO single-path"
LABEL_DUAL = "RoboReflex dual-path"
BADGE_SLOWMO = "SLOW-MO ×0.15"

with open(PROJECT_ROOT / "src" / "scenarios" / "ball_catch_easy.yaml", encoding="utf-8") as f:
    SCENARIO = yaml.safe_load(f)
SCENARIO["launch"]["delay_range"] = [2.5, 3.5]   # 拉长空闲期（同 record_idle_compare）

FONT = ImageFont.load_default(size=15)
FONT_BIG = ImageFont.load_default(size=26)


def _lower_priority():
    """另一个进程在跑训练，降到 BELOW_NORMAL 减少争抢（失败也无妨）。"""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass


def centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def find_ball_paddle(arr):
    img = arr.astype(np.int16)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    ball = centroid((r > 180) & (g < 120) & (b < 120))          # 红球
    paddle = centroid((b > 150) & (g > 120) & (r < 120))        # 蓝挡板
    return ball, paddle


def make_meta(base, engaged=False):
    ev = base._events
    stim = ev.get("stim", {}).get("sim") if "stim" in ev else None
    fa = ev.get("first_action", {}).get("sim") if "first_action" in ev else None
    return {"t": float(base.data.time), "launched": base.ball_launched,
            "stim": stim, "fa": fa, "engaged": bool(engaged),
            "outcome": base._done_type}


def run_pair(seed, fast, render):
    """同种子同步跑 PPO 单通路与双通路，返回逐帧画面与事件元数据。"""
    mode = "rgb_array" if render else None
    base_p = PaddleCatchEnv(scenario=SCENARIO, seed=seed, render_mode=mode)
    base_d = PaddleCatchEnv(scenario=SCENARIO, seed=seed, render_mode=mode)
    slow = CruisePlanner(v_max=base_d.v_max, x_limit=base_d.x_limit)
    dual = DualPathEnv(base_d, slow, fast)
    obs_p, _ = base_p.reset(seed=seed)
    dual.reset(seed=seed)
    done_p = done_d = False
    step_i = 0
    frames_p, frames_d, metas_p, metas_d = [], [], [], []
    info_p = info_d = {}
    while not (done_p and done_d):
        if not done_p:
            obs_p, _, term_p, _, info_p = base_p.step(fast.act(obs_p, base_p.ball_launched))
            done_p = term_p
        if not done_d:
            _, _, term_d, _, info_d = dual.step()
            done_d = term_d
        step_i += 1
        # 终局步也要捕获（否则双环境同时在奇数步结束时结果帧丢失）
        if step_i % STRIDE == 0 or (done_p and done_d):
            if render:
                frames_p.append(base_p.render())
                frames_d.append(base_d.render())
            metas_p.append(make_meta(base_p))
            metas_d.append(make_meta(base_d, dual.arbiter.engaged))
    base_p.close()
    base_d.close()
    return {"frames_p": frames_p, "frames_d": frames_d,
            "metas_p": metas_p, "metas_d": metas_d,
            "events_p": info_p.get("events", {}), "events_d": info_d.get("events", {}),
            "outcome_p": base_p._done_type, "outcome_d": base_d._done_type}


def pick_seed(fast, candidates=range(1, 31)):
    """优先选 双通路接住 & PPO 误触发 的种子；同分取 PPO 空闲期抖动更大者。"""
    best, best_key = None, None
    for seed in candidates:
        r = run_pair(seed, fast, render=False)
        score = 0
        if r["outcome_d"] == "caught":
            score += 2
        if r["outcome_p"] in ("failed", "timeout"):
            score += 2
        if r["events_p"].get("false_trigger"):
            score += 1
        dev = float(r["events_p"].get("prestim_max_dev", 0.0))
        key = (score, dev)
        print(f"  seed {seed}: ppo={r['outcome_p']} dual={r['outcome_d']} "
              f"ppo_false_trigger={r['events_p'].get('false_trigger')} "
              f"prestim_dev={dev:.3f} score={score}", flush=True)
        if best_key is None or key > best_key:
            best, best_key = seed, key
    return best


def draw_panel(arr, meta, label, is_dual, trail):
    ball, paddle = find_ball_paddle(arr)
    im = Image.fromarray(arr).convert("RGBA")

    # 残影拖尾：越新的残影越亮
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    n = len(trail)
    for i, (tx, ty) in enumerate(trail):
        a = int(15 + 95 * (i + 1) / n)
        od.ellipse([tx - 30, ty - 5, tx + 30, ty + 5], fill=(120, 220, 255, a))
    im = Image.alpha_composite(im, overlay).convert("RGB")

    d = ImageDraw.Draw(im)
    # 系统标签（左上）
    lw = d.textlength(label, font=FONT)
    d.rectangle([6, 4, 6 + lw + 12, 26], fill=(15, 15, 15))
    d.text((12, 8), label, font=FONT, fill=(240, 240, 240))

    # HUD 延迟条（右上）：发球后实时计数，首动后定格；双通路接管时绿框亮起
    stim, fa, t = meta["stim"], meta["fa"], meta["t"]
    if stim is None:
        rt_text, rt_color = "RT -- ms", (170, 170, 170)
    elif fa is None or t < fa:
        rt_text, rt_color = f"RT {(t - stim) * 1000:3.0f} ms", (255, 214, 90)
    else:
        rt_text, rt_color = f"RT {(fa - stim) * 1000:3.0f} ms", (255, 255, 255)
    tw = d.textlength(rt_text, font=FONT)
    x0, y0 = PANEL_W - 10 - tw - 10, 4
    d.rectangle([x0, y0, x0 + tw + 12, y0 + 22], fill=(15, 15, 15))
    d.text((x0 + 6, y0 + 4), rt_text, font=FONT, fill=rt_color)
    if is_dual and meta["engaged"]:
        d.rectangle([x0 - 2, y0 - 2, x0 + tw + 14, y0 + 24],
                    outline=(90, 255, 130), width=2)

    # 文字标注：球=Stimulus（黄箭头），挡板=Reflex paddle（绿）
    if ball:
        x, y = ball
        d.line([(x, y - 8), (x, y - 40)], fill=(255, 255, 0), width=2)
        d.polygon([(x - 5, y - 40), (x + 5, y - 40), (x, y - 32)], fill=(255, 255, 0))
        tx = int(min(max(x + 8, 2), PANEL_W - d.textlength("Stimulus", font=FONT) - 2))
        d.text((tx, y - 50), "Stimulus", font=FONT, fill=(255, 255, 0))
    if paddle:
        x, y = paddle
        d.line([(x, y + 8), (x, y + 36)], fill=(120, 255, 120), width=2)
        d.polygon([(x - 5, y + 36), (x + 5, y + 36), (x, y + 28)], fill=(120, 255, 120))
        tx = int(min(max(x + 8, 2), PANEL_W - d.textlength("Reflex paddle", font=FONT) - 2))
        d.text((tx, y + 30), "Reflex paddle", font=FONT, fill=(120, 255, 120))

    # 终局结果（底部居中）
    if meta["outcome"]:
        text, color = (("CAUGHT", (90, 255, 130)) if meta["outcome"] == "caught"
                       else ("MISSED", (255, 90, 90)))
        bw = d.textlength(text, font=FONT_BIG)
        bx, by = (PANEL_W - bw) / 2, PANEL_H - 52
        d.rectangle([bx - 10, by - 6, bx + bw + 10, by + 34], fill=(15, 15, 15))
        d.text((bx, by), text, font=FONT_BIG, fill=color)

    if paddle:
        trail.append(paddle)
    return im


def compose(im_l, im_r, badge=None):
    canvas = Image.new("RGB", (W, H))
    canvas.paste(im_l, (0, 0))
    canvas.paste(im_r, (PANEL_W, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([PANEL_W - 2, 0, PANEL_W + 1, H], fill=(45, 45, 45))
    if badge:
        bw = d.textlength(badge, font=FONT)
        d.rectangle([(W - bw) / 2 - 10, 6, (W + bw) / 2 + 10, 30], fill=(15, 15, 15))
        d.text(((W - bw) / 2, 10), badge, font=FONT, fill=(255, 214, 90))
    return canvas


def render_window(data, indices, badge=None):
    """渲染一段帧索引（可越界=定格末帧），拖尾从窗口起点前 TRAIL_LEN 帧预热。"""
    frames_p, frames_d = data["frames_p"], data["frames_d"]
    metas_p, metas_d = data["metas_p"], data["metas_d"]
    last = len(frames_p) - 1
    trail_p, trail_d = deque(maxlen=TRAIL_LEN), deque(maxlen=TRAIL_LEN)
    for i in range(max(0, indices[0] - TRAIL_LEN), min(indices[0], last + 1)):
        _, p = find_ball_paddle(frames_p[i])
        if p:
            trail_p.append(p)
        _, p = find_ball_paddle(frames_d[i])
        if p:
            trail_d.append(p)
    out = []
    for i in indices:
        i = min(i, last)
        left = draw_panel(frames_p[i], metas_p[i], LABEL_PPO, False, trail_p)
        right = draw_panel(frames_d[i], metas_d[i], LABEL_DUAL, True, trail_d)
        out.append(compose(left, right, badge))
    return out


def save_gif(frames, durations_ms, colors):
    mosaic = Image.new("RGB", (W, H * len(frames[:: max(1, len(frames) // 12)])))
    for k, f in enumerate(frames[:: max(1, len(frames) // 12)]):
        mosaic.paste(f, (0, H * k))
    palette = mosaic.quantize(colors=colors, method=Image.MEDIANCUT,
                              dither=Image.Dither.NONE)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quant[0].save(OUT_GIF, save_all=True, append_images=quant[1:],
                  duration=durations_ms, loop=0, optimize=True)
    return OUT_GIF.stat().st_size


def selfcheck_strip(n_frames, durations_ms):
    frames = imageio.mimread(OUT_GIF)
    picks = [1, n_frames // 6, n_frames // 3, n_frames // 2,
             (2 * n_frames) // 3, n_frames - 1]
    tiles = [Image.fromarray(frames[p][:, :, :3]).resize((W * 2 // 5, H * 2 // 5))
             for p in picks]
    strip = Image.new("RGB", (tiles[0].width * len(tiles), tiles[0].height))
    for k, t in enumerate(tiles):
        strip.paste(t, (t.width * k, 0))
    strip.save(OUT_STRIP)
    return picks


def main():
    _lower_priority()
    fast = load_ppo_policy(str(PROJECT_ROOT / "runs" / "ppo_alpha"))

    print("[scan] 无渲染扫种子，选对比最强的一个...", flush=True)
    seed = pick_seed(fast)
    print(f"[scan] 选定 seed={seed}", flush=True)

    print("[record] 同种子同步录制双环境...", flush=True)
    data = run_pair(seed, fast, render=True)
    metas_d, metas_p = data["metas_d"], data["metas_p"]
    n_cap = len(metas_d)
    S = next(i for i, m in enumerate(metas_d) if m["launched"])
    catch_d = next(i for i, m in enumerate(metas_d) if m["outcome"] is not None)
    done_p = next((i for i, m in enumerate(metas_p) if m["outcome"] is not None),
                  n_cap - 1)
    print(f"[record] 捕获 {n_cap} 帧：发球帧 {S}，双通路结果帧 {catch_d}，"
          f"PPO 结果帧 {done_p}", flush=True)

    # 时间轴：主段 = 短空闲 + 飞行 + 结果定格；慢放段 = 发球前 0.3s -> 接住
    lead = round(SLOWMO_LEAD_S * FPS)
    sm_start, sm_end = max(0, S - lead), catch_d
    n_sm = sm_end - sm_start + 1
    sm_dur = 1.0 / FPS / SLOWMO_RATE
    PAD = 5
    main_end = max(catch_d, done_p) + PAD
    n_flight_main = main_end - S + 1
    idle_budget = TIME_BUDGET_S - n_sm * sm_dur - n_flight_main / FPS
    idle_keep = max(8, min(20, int(idle_budget * FPS)))
    main_start = max(0, S - idle_keep)

    main_frames = render_window(data, list(range(main_start, main_end + 1)))
    sm_frames = render_window(data, list(range(sm_start, sm_end + 1)), BADGE_SLOWMO)
    frames = main_frames + sm_frames
    durations = [40] * len(main_frames) + [round(1000 / FPS / SLOWMO_RATE)] * len(sm_frames)
    total_s = sum(durations) / 1000
    print(f"[edit] 主段 {len(main_frames)} 帧 + 慢放 {len(sm_frames)} 帧，"
          f"总时长 {total_s:.2f}s", flush=True)

    size = None
    for colors in (128, 96, 64):
        size = save_gif(frames, durations, colors)
        print(f"[save] {colors} 色 -> {size / 1024 / 1024:.2f} MB", flush=True)
        if size <= SIZE_LIMIT:
            break
    if size > SIZE_LIMIT:
        raise SystemExit(f"GIF 超体积：{size / 1024 / 1024:.2f} MB > 5 MB")

    picks = selfcheck_strip(len(frames), durations)
    rt_p = next((m for m in metas_p if m["fa"] is not None), None)
    rt_d = next((m for m in metas_d if m["fa"] is not None), None)
    n_engaged = sum(1 for m in metas_d if m["engaged"])
    print(f"[done] {OUT_GIF}  {size / 1024 / 1024:.2f} MB, {total_s:.2f}s, "
          f"{len(frames)} 帧 @ {W}x{H}", flush=True)
    print(f"[done] PPO RT={((rt_p['fa'] - rt_p['stim']) * 1000):.0f}ms "
          f"dual RT={((rt_d['fa'] - rt_d['stim']) * 1000):.0f}ms "
          f"双通路接管帧数={n_engaged}/{n_cap}", flush=True)
    print(f"[done] 自检条 {OUT_STRIP}（抽帧 {picks}）", flush=True)


if __name__ == "__main__":
    main()
