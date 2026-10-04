"""REWARD.md 消融矩阵评测（M4）：B(λ=0) / C(λ=0.5,1.0,2.0) / D(α动态) 单通路对比。

每组 200 次试验同种子，输出 assets/ablation.json，供 make_figures.py 画
"成功率-延迟 p95"帕累托散点图。叙事验证：固定 λ 每档是帕累托前沿上一个
固定点，α 动态自动收敛到最优点。

用法：python -m src.bench.ablation --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..latency.profiler import LatencyProfiler
from ..reflex.env import PaddleCatchEnv
from .baseline import run_episode
from .compare import evaluate_group
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ABLATION_GROUPS = [
    ("B lambda=0", "ppo_lambda0"),
    ("C lambda=0.5", "ppo_fixed_0.5"),
    ("C lambda=1.0", "ppo_fixed_1.0"),
    ("C lambda=2.0", "ppo_fixed_2.0"),
    ("D alpha-dyn", "ppo_alpha"),
]


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4: fixed-lambda ablation (B/C/D)")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--json-out", default=str(PROJECT_ROOT / "assets" / "ablation.json"))
    args = ap.parse_args()

    env = PaddleCatchEnv(seed=args.seed)
    rows = []
    for label, run_name in ABLATION_GROUPS:
        run_dir = Path(args.runs) / run_name
        if not (run_dir / "final_model.zip").exists():
            print(f"skip {label}: {run_dir} not found", flush=True)
            continue
        print(f"evaluating {label} ...", flush=True)
        policy = load_ppo_policy(run_dir)
        prof = evaluate_group(env, policy, args.trials, args.seed, label)
        rows.append({"label": label, "run": run_name, "summary": prof.summary(),
                     "trials": prof.trials})
    env.close()

    print(f"\n{'group':<14}{'SR':>7}{'react p50(ms)':>15}{'react p95(ms)':>15}")
    print("-" * 51)
    for r in rows:
        s, rs = r["summary"], r["summary"]["reaction_sim_s"]
        print(f"{r['label']:<14}{s['success_rate']:>7.3f}"
              f"{rs['p50'] * 1e3:>15.1f}{rs['p95'] * 1e3:>15.1f}")

    Path(args.json_out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
