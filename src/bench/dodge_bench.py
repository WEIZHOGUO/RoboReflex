"""第二场景评测（M4）：避障 dodge_easy——验证"场景即配置"+ 软过渡真实触发。

两组：手写避障基线 / 机映双通路（慢巡航 + 仲裁器 + 避障 α 策略快通路）。
关键产出：
- 双通路成功率 / 误触发率 / 延迟剖面
- handover 实证：仲裁器交还次数、blend 步出现的试验占比（球掠过=威胁解除）
输出 assets/dodge.json，供 make_figures.py 画双场景对比图。

用法：python -m src.bench.dodge_bench --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .baseline import DodgeBaseline, run_episode
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DODGE_SCENARIO = PROJECT_ROOT / "src" / "scenarios" / "dodge_easy.yaml"


def run_dodge_bench(n_trials: int, seed0: int, dodge_run,
                    verbose: bool = True) -> dict:
    rows = []

    # 组 1：手写避障基线（单通路）
    env = PaddleCatchEnv(scenario_path=DODGE_SCENARIO, seed=seed0)
    baseline = DodgeBaseline(v_max=env.v_max, catch_z=env.catch_z,
                             x_limit=env.x_limit, dodge_radius=env.dodge_radius)
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(run_episode(env, baseline, seed=seed0 + i))
    rows.append({"label": "dodge baseline", "summary": prof.summary(),
                 "trials": prof.trials})
    env.close()
    if verbose:
        s = rows[-1]["summary"]
        print(f"  dodge baseline: SR={s['success_rate']:.3f} "
              f"FTR={s['false_trigger_rate']:.3f}", flush=True)

    # 组 2：双通路（慢巡航 + 仲裁器 + 避障 α 策略）
    base = PaddleCatchEnv(scenario_path=DODGE_SCENARIO, seed=seed0)
    env = DualPathEnv(base, CruisePlanner(v_max=base.v_max, x_limit=base.x_limit),
                      load_ppo_policy(dodge_run))
    prof = LatencyProfiler()
    n_handover = 0
    n_blend_steps = 0
    for i in range(n_trials):
        events = run_dualpath_episode(env, seed=seed0 + i)
        prof.record(events)
        seq = events.get("switch_sequence", [])
        if events.get("arbiter", {}).get("disengagements", 0) > 0:
            n_handover += 1
            n_blend_steps += seq.count("blend")
    rows.append({"label": "dodge dual-path", "summary": prof.summary(),
                 "trials": prof.trials})
    env.close()

    handover = {
        "trials_with_disengagement": n_handover,
        "handover_rate": n_handover / n_trials,
        "blend_steps_total": n_blend_steps,
    }
    if verbose:
        s = rows[-1]["summary"]
        rs = s["reaction_sim_s"]
        print(f"  dodge dual-path: SR={s['success_rate']:.3f} "
              f"FTR={s['false_trigger_rate']:.3f} "
              f"p95={rs['p95'] * 1e3:.1f}ms", flush=True)
        print(f"  handover: {n_handover}/{n_trials} trials with soft handover, "
              f"{n_blend_steps} blend steps", flush=True)
    return {"scenario": "dodge_easy", "groups": rows, "handover": handover}


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4: dodge scenario benchmark")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--dodge-run", default=str(PROJECT_ROOT / "runs" / "ppo_dodge_alpha"))
    ap.add_argument("--json-out", default=str(PROJECT_ROOT / "assets" / "dodge.json"))
    args = ap.parse_args()

    result = run_dodge_bench(args.trials, args.seed, args.dodge_run)
    Path(args.json_out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
