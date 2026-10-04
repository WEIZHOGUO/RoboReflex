"""难度课程（M4）：球速扫描 2→8 m/s，检验双通路在更难分布下的衰减。

每档球速把场景 speed_range 钉成单点，跑手写基线与双通路（复用 M2 α 策略，
不重训——考验分布外泛化）各 n 次试验，输出"成功率 vs 球速"和
"延迟 p95 vs 球速"数据到 assets/speed_sweep.json。

用法：python -m src.bench.speed_sweep --trials 100
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .baseline import BallisticBaseline, run_episode
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATCH_SCENARIO = PROJECT_ROOT / "src" / "scenarios" / "ball_catch_easy.yaml"
SPEED_LEVELS = [2.0, 3.0, 4.0, 5.5, 7.0, 8.0]


def _scenario_at_speed(speed: float) -> dict:
    with open(CATCH_SCENARIO, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["launch"]["speed_range"] = [speed, speed]
    return cfg


def run_sweep(levels=None, n_trials: int = 100, seed0: int = 30_000,
              alpha_run=None, verbose: bool = True) -> dict:
    levels = list(levels or SPEED_LEVELS)
    fast = load_ppo_policy(alpha_run or (PROJECT_ROOT / "runs" / "ppo_alpha"))
    rows = []
    for v in levels:
        cfg = _scenario_at_speed(v)
        row = {"speed": v}

        env = PaddleCatchEnv(scenario=cfg, seed=seed0)
        baseline = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                                     x_limit=env.x_limit)
        prof = LatencyProfiler()
        for i in range(n_trials):
            prof.record(run_episode(env, baseline, seed=seed0 + i))
        row["baseline"] = prof.summary()
        env.close()

        base = PaddleCatchEnv(scenario=cfg, seed=seed0)
        env = DualPathEnv(base, CruisePlanner(v_max=base.v_max,
                                              x_limit=base.x_limit), fast)
        prof = LatencyProfiler()
        for i in range(n_trials):
            prof.record(run_dualpath_episode(env, seed=seed0 + i))
        row["dual_path"] = prof.summary()
        env.close()

        rows.append(row)
        if verbose:
            b, d = row["baseline"], row["dual_path"]
            print(f"speed={v:>4.1f} m/s: baseline SR={b['success_rate']:.3f} "
                  f"p95={b['reaction_sim_s']['p95'] * 1e3:.1f}ms | "
                  f"dual SR={d['success_rate']:.3f} "
                  f"p95={d['reaction_sim_s']['p95'] * 1e3:.1f}ms "
                  f"FTR={d['false_trigger_rate']:.3f}", flush=True)
    return {"levels": levels, "n_trials": n_trials, "seed0": seed0, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4: ball speed sweep")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=30_000)
    ap.add_argument("--alpha-run", default=str(PROJECT_ROOT / "runs" / "ppo_alpha"))
    ap.add_argument("--json-out",
                    default=str(PROJECT_ROOT / "assets" / "speed_sweep.json"))
    args = ap.parse_args()

    result = run_sweep(n_trials=args.trials, seed0=args.seed,
                       alpha_run=args.alpha_run)
    Path(args.json_out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
