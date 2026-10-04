"""多种子评测（M4）：α动态双通路 × 5 个训练种子，报均值±标准差。

种子 42 复用 M2 的 runs/ppo_alpha，其余为 runs/ppo_alpha_s<seed>。
输出 assets/multiseed.json，供 make_figures.py 画误差棒图。

用法：python -m src.bench.multiseed --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [42, 123, 456, 789, 1024]


def run_dir_for(seed: int) -> str:
    return "ppo_alpha" if seed == 42 else f"ppo_alpha_s{seed}"


def eval_dualpath_seed(run_dir, n_trials: int, eval_seed: int) -> dict:
    base = PaddleCatchEnv(seed=eval_seed)
    env = DualPathEnv(base, CruisePlanner(v_max=base.v_max, x_limit=base.x_limit),
                      load_ppo_policy(run_dir))
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(run_dualpath_episode(env, seed=eval_seed + i))
    env.close()
    return prof.summary()


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4: multi-seed dual-path eval")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000, help="eval seed base")
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--json-out", default=str(PROJECT_ROOT / "assets" / "multiseed.json"))
    args = ap.parse_args()

    per_seed = []
    for s in SEEDS:
        run_dir = Path(args.runs) / run_dir_for(s)
        if not (run_dir / "final_model.zip").exists():
            print(f"skip seed {s}: {run_dir} not found", flush=True)
            continue
        print(f"evaluating dual-path seed={s} ...", flush=True)
        summ = eval_dualpath_seed(run_dir, args.trials, args.seed)
        per_seed.append({"train_seed": s, "summary": summ})
        rs = summ["reaction_sim_s"]
        print(f"  seed={s}: SR={summ['success_rate']:.3f} "
              f"FTR={summ['false_trigger_rate']:.3f} "
              f"p95={rs['p95'] * 1e3:.1f}ms", flush=True)

    metrics = {
        "success_rate": [p["summary"]["success_rate"] for p in per_seed],
        "false_trigger_rate": [p["summary"]["false_trigger_rate"] for p in per_seed],
        "react_p50_ms": [p["summary"]["reaction_sim_s"]["p50"] * 1e3 for p in per_seed],
        "react_p95_ms": [p["summary"]["reaction_sim_s"]["p95"] * 1e3 for p in per_seed],
    }
    aggregate = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                     "values": v} for k, v in metrics.items()}

    print("\n===== multi-seed (dual-path, mean +/- std over %d seeds) =====" % len(per_seed))
    for k, a in aggregate.items():
        print(f"  {k:<20} {a['mean']:.3f} +/- {a['std']:.3f}")

    payload = {"seeds": [p["train_seed"] for p in per_seed],
               "per_seed": per_seed, "aggregate": aggregate}
    Path(args.json_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
