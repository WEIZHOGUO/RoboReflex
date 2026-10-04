"""录制策略接球 GIF（与手写基线 demo.gif 并列用）。

用法：
  python -m src.bench.record_gif --system ppo  --run runs/ppo_alpha --out assets/demo_ppo.gif
  python -m src.bench.record_gif --system dual --run runs/ppo_alpha --out assets/demo_dualpath.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description="Record policy catching balls to GIF")
    ap.add_argument("--system", choices=["ppo", "dual"], default="ppo")
    ap.add_argument("--run", default=str(PROJECT_ROOT / "runs" / "ppo_alpha"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--trials", type=int, default=3, help="number of trials to record")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stride", type=int, default=2, help="frame stride")
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROJECT_ROOT / "assets" / (
        "demo_dualpath.gif" if args.system == "dual" else "demo_ppo.gif")

    base = PaddleCatchEnv(seed=args.seed, render_mode="rgb_array")
    fast = load_ppo_policy(args.run)
    if args.system == "dual":
        slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit)
        env = DualPathEnv(base, slow, fast)
        act = None  # 双通路内部 mux，无需外部动作
    else:
        env = base
        act = lambda obs: fast.act(obs, base.ball_launched)  # noqa: E731

    profiler = LatencyProfiler()
    frames = []
    for i in range(args.trials):
        obs, _ = env.reset(seed=args.seed + i)
        step_i = 0
        while True:
            if act is not None:
                obs, _, terminated, truncated, info = env.step(act(obs))
            else:
                obs, _, terminated, truncated, info = env.step()
            if step_i % args.stride == 0:
                frames.append(env.render())
            step_i += 1
            if terminated or truncated:
                break
        trial = profiler.record(info.get("events", {}))
        reaction = trial["reaction_sim_s"]
        arb = trial.get("arbiter")
        extra = f" arbiter={arb}" if arb else ""
        print(f"trial {i + 1}: outcome={trial['outcome']} "
              f"reaction={None if reaction is None else round(reaction * 1e3, 1)}ms"
              f"{extra}", flush=True)
    env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    fps = base.control_hz / args.stride
    imageio.mimsave(out, frames, duration=1.0 / fps)
    s = profiler.summary()
    print(f"GIF saved: {out} ({len(frames)} frames, {fps:.0f}fps), "
          f"SR={s['success_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
