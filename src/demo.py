"""可视化 demo：手写弹道基线接球 + 实时成功率/延迟统计。

用法：
  python -m src.demo                    # 无头跑 100 次试验，打印统计
  python -m src.demo --viewer           # MuJoCo 交互窗口实时观看
  python -m src.demo --record           # 离屏渲染前若干次试验，存 assets/demo.gif
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bench.baseline import BallisticBaseline
from .latency.profiler import LatencyProfiler
from .reflex.env import PaddleCatchEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_env(args, render_mode=None) -> PaddleCatchEnv:
    return PaddleCatchEnv(scenario_path=args.scenario, control_hz=args.control_hz,
                          seed=args.seed, render_mode=render_mode)


def make_policy(env: PaddleCatchEnv) -> BallisticBaseline:
    return BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z, x_limit=env.x_limit)


def run_trials(env, policy, profiler: LatencyProfiler, n_trials: int, seed: int,
               record_gif: Path | None = None, record_trials: int = 3,
               record_stride: int = 2):
    frames = []
    for i in range(n_trials):
        recording = record_gif is not None and i < record_trials
        obs, _ = env.reset(seed=seed + i)
        step_i = 0
        while True:
            action = policy.act(obs, env.ball_launched)
            obs, _, terminated, truncated, info = env.step(action)
            if recording and step_i % record_stride == 0:
                frames.append(env.render())
            step_i += 1
            if terminated or truncated:
                break
        profiler.record(info.get("events", {}))
        if (i + 1) % 10 == 0 or i + 1 == n_trials:
            s = profiler.summary()
            r = s["reaction_sim_s"] or {}
            print(f"[{i + 1:>4}/{n_trials}] 成功率 {s['success_rate']:.1%} "
                  f"误触发率 {s['false_trigger_rate']:.1%} "
                  f"反应延迟 p50={r.get('p50', float('nan')) * 1e3:.1f}ms "
                  f"p95={r.get('p95', float('nan')) * 1e3:.1f}ms")

    if record_gif is not None and frames:
        import imageio.v2 as imageio
        record_gif.parent.mkdir(parents=True, exist_ok=True)
        fps = env.control_hz / record_stride
        imageio.mimsave(record_gif, frames, duration=1.0 / fps)
        print(f"GIF 已保存: {record_gif}（{len(frames)} 帧，{fps:.0f}fps）")


def run_viewer(env, policy, n_trials: int, seed: int):
    import mujoco.viewer
    profiler = LatencyProfiler()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(n_trials):
            obs, _ = env.reset(seed=seed + i)
            viewer.sync()
            while viewer.is_running():
                t0 = time.perf_counter()
                action = policy.act(obs, env.ball_launched)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                if terminated or truncated:
                    profiler.record(info.get("events", {}))
                    print(json.dumps(profiler.summary(), ensure_ascii=False, default=str))
                    break
                dt = 1.0 / env.control_hz - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)
            if not viewer.is_running():
                break


def main():
    ap = argparse.ArgumentParser(description="机映 RoboReflex M1 demo：手写弹道基线接球")
    ap.add_argument("--scenario", default=str(PROJECT_ROOT / "src" / "scenarios" / "ball_catch_easy.yaml"))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control-hz", type=int, default=50)
    ap.add_argument("--record", action="store_true", help="离屏渲染并存 GIF 到 assets/demo.gif")
    ap.add_argument("--record-trials", type=int, default=3, help="GIF 收录前几次试验")
    ap.add_argument("--viewer", action="store_true", help="打开 MuJoCo 交互窗口")
    ap.add_argument("--json-out", default=None, help="延迟剖面 JSON 输出路径")
    args = ap.parse_args()

    if args.viewer:
        env = make_env(args)
        run_viewer(env, make_policy(env), args.trials, args.seed)
        env.close()
        return

    env = make_env(args, render_mode="rgb_array" if args.record else None)
    profiler = LatencyProfiler()
    gif_path = PROJECT_ROOT / "assets" / "demo.gif" if args.record else None
    run_trials(env, make_policy(env), profiler, args.trials, args.seed,
               record_gif=gif_path, record_trials=args.record_trials)
    env.close()

    summary = profiler.summary()
    print("\n===== 最终统计 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        profiler.save(args.json_out)
        print(f"延迟剖面已保存: {args.json_out}")


if __name__ == "__main__":
    main()
