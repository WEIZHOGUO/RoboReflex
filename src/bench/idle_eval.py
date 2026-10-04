"""空闲惩罚对照组评测（M4.5，审稿人反击①）：奖励工程能否压误触发。

组：PPO α动态（无空闲惩罚，参照）/ +idle β=0.05 / +idle β=0.10，单通路，
各 200 次试验同种子。误触发口径 = 环境 prestim 位移（单通路无仲裁器，
反射输出直连执行器）。输出 assets/idle_penalty.json。

用法：python -m src.bench.idle_eval --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..latency.profiler import LatencyProfiler
from ..reflex.env import PaddleCatchEnv
from .compare import evaluate_group
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUPS = [
    ("alpha-dyn (ref)", "ppo_alpha"),
    ("idle beta=0.05", "ppo_idle_0.05"),
    ("idle beta=0.10", "ppo_idle_0.10"),
]


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4.5: idle-penalty ablation eval")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--json-out",
                    default=str(PROJECT_ROOT / "assets" / "idle_penalty.json"))
    args = ap.parse_args()

    env = PaddleCatchEnv(seed=args.seed)
    rows = []
    for label, run_name in GROUPS:
        run_dir = Path(args.runs) / run_name
        if not (run_dir / "final_model.zip").exists():
            print(f"skip {label}: {run_dir} not found", flush=True)
            continue
        print(f"evaluating {label} ...", flush=True)
        prof = evaluate_group(env, load_ppo_policy(run_dir),
                              args.trials, args.seed, label)
        rows.append({"label": label, "run": run_name, "summary": prof.summary(),
                     "trials": prof.trials})
    env.close()

    head = f"\n{'group':<18}{'SR':>7}{'FTR':>7}{'p50(ms)':>9}{'p95(ms)':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        s, rs = r["summary"], r["summary"]["reaction_sim_s"]
        print(f"{r['label']:<18}{s['success_rate']:>7.3f}"
              f"{s['false_trigger_rate']:>7.3f}"
              f"{rs['p50'] * 1e3:>9.1f}{rs['p95'] * 1e3:>9.1f}")

    Path(args.json_out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
