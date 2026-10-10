"""录制 ReflexSuite-mini 三场景 demo GIF（默认双通路方案）。

用法：
  python -m src.bench.record_reflexsuite_gif --task ball_catch
  python -m src.bench.record_reflexsuite_gif --task whack_a_mole --system ppo
  python -m src.bench.record_reflexsuite_gif --task rolling_ball --system baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv
from ..reflex.slowpath import CruisePlanner
from .policies import load_ppo_policy
from .reflexsuite_bench import GROUP_LABELS, _baseline_for, make_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description="Record ReflexSuite-mini demo GIFs")
    ap.add_argument("--task", choices=["ball_catch", "whack_a_mole", "rolling_ball"],
                    required=True)
    ap.add_argument("--system", choices=["baseline", "ppo", "dual"], default="dual")
    ap.add_argument("--run", default=None,
                    help="PPO run dir (default runs/ppo_rs_<task>_alpha)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    run = Path(args.run) if args.run else PROJECT_ROOT / "runs" / f"ppo_rs_{args.task}_alpha"
    out = Path(args.out) if args.out else PROJECT_ROOT / "assets" / (
        f"demo_reflexsuite_{args.task}.gif" if args.system == "dual"
        else f"demo_reflexsuite_{args.task}_{args.system}.gif")

    base = make_env(args.task, args.seed)
    base.render_mode = "rgb_array"
    if args.system == "dual":
        slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit)
        env = DualPathEnv(base, slow, load_ppo_policy(run))
        act = None
    elif args.system == "ppo":
        pol = load_ppo_policy(run)
        env = base
        act = lambda obs: pol.act(obs, base.ball_launched)  # noqa: E731
    else:
        pol = _baseline_for(base, args.task)
        env = base
        act = lambda obs: pol.act(obs, base.ball_launched)  # noqa: E731

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
        print(f"trial {i + 1}: outcome={trial['outcome']} "
              f"reaction={None if reaction is None else round(reaction * 1e3, 1)}ms",
              flush=True)
    env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    fps = base.control_hz / args.stride
    imageio.mimsave(out, frames, duration=1.0 / fps)
    s = profiler.summary()
    print(f"GIF saved: {out} ({len(frames)} frames, {fps:.0f}fps), "
          f"system={GROUP_LABELS.get(args.system, args.system)} "
          f"SR={s['success_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
