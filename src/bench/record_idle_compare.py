"""录制"空闲期对比"GIF：拉长无球等待期，凸显误触发差异（README 首图用）。

  python -m src.bench.record_idle_compare
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import yaml

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 基于 easy 场景，只把发球延迟拉长到 2.5~3.5s（空闲期成为主角）
with open(PROJECT_ROOT / "src" / "scenarios" / "ball_catch_easy.yaml", encoding="utf-8") as f:
    SCENARIO = yaml.safe_load(f)
SCENARIO["launch"]["delay_range"] = [2.5, 3.5]


def record(system: str, out: Path, trials: int = 3, seed: int = 7, stride: int = 2):
    base = PaddleCatchEnv(scenario=SCENARIO, seed=seed, render_mode="rgb_array")
    fast = load_ppo_policy(str(PROJECT_ROOT / "runs" / "ppo_alpha"))
    if system == "dual":
        slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit)
        env = DualPathEnv(base, slow, fast)
        act = None
    else:
        env = base
        act = lambda obs: fast.act(obs, base.ball_launched)  # noqa: E731

    profiler = LatencyProfiler()
    frames = []
    for i in range(trials):
        obs, _ = env.reset(seed=seed + i)
        step_i = 0
        while True:
            if act is not None:
                obs, _, terminated, truncated, info = env.step(act(obs))
            else:
                obs, _, terminated, truncated, info = env.step()
            if step_i % stride == 0:
                frames.append(env.render())
            step_i += 1
            if terminated or truncated:
                break
        trial = profiler.record(info.get("events", {}))
        print(f"[{system}] trial {i+1}: outcome={trial['outcome']}", flush=True)
    env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    fps = base.control_hz / stride
    imageio.mimsave(out, frames, duration=1.0 / fps)
    s = profiler.summary()
    print(f"[{system}] saved {out} ({len(frames)} frames), SR={s['success_rate']:.2f} "
          f"FTR={s['false_trigger_rate']:.2f}", flush=True)


if __name__ == "__main__":
    record("ppo", PROJECT_ROOT / "assets" / "idle_ppo.gif")
    record("dual", PROJECT_ROOT / "assets" / "idle_dualpath.gif")
