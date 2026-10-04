"""四组对比跑批：手写基线 / PPO(λ=0) / PPO(α动态) / 机映双通路。

每组 n 次试验（同一组随机种子序列，公平对比），输出：
- 控制台对比表（成功率、反应延迟 p50/p95、误触发率；双通路附五段式账本）
- assets/bench_compare_v2.png：延迟剖面直方图 + 成功率/误触发率柱状图
- assets/bench_compare_v2.json：原始 trial 数据 + 汇总

双通路组 = 慢通路巡航（5Hz）+ 仲裁器（滞回+驻留）+ M2 的 α 策略做快通路。
误触发口径（ARCHITECTURE 九.3）：乱拉警报的是仲裁器——双通路组 FTR =
仲裁器误接管率；单通路组 FTR = 无球期位移（反射输出直连执行器，抖动即误报）。

用法：
  python -m src.bench.compare --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .baseline import BallisticBaseline, run_episode
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_group(env: PaddleCatchEnv, policy, n_trials: int, seed0: int,
                   label: str) -> LatencyProfiler:
    prof = LatencyProfiler()
    for i in range(n_trials):
        events = run_episode(env, policy, seed=seed0 + i)
        prof.record(events)
        if (i + 1) % 50 == 0 or i + 1 == n_trials:
            print(f"  [{label}] {i + 1}/{n_trials} done", flush=True)
    return prof


def evaluate_dualpath(n_trials: int, seed0: int, alpha_run: str,
                      label: str) -> LatencyProfiler:
    """机映双通路：慢巡航 + 仲裁器 + M2 α策略（快通路）。"""
    base = PaddleCatchEnv(seed=seed0)
    slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit)
    fast = load_ppo_policy(alpha_run)
    env = DualPathEnv(base, slow, fast)
    prof = LatencyProfiler()
    for i in range(n_trials):
        events = run_dualpath_episode(env, seed=seed0 + i)
        prof.record(events)
        if (i + 1) % 50 == 0 or i + 1 == n_trials:
            print(f"  [{label}] {i + 1}/{n_trials} done", flush=True)
    env.close()
    return prof


def fmt_ms(stats: dict | None, key: str) -> str:
    if not stats:
        return "  n/a "
    return f"{stats[key] * 1e3:6.1f}"


def print_table(rows: list[dict]) -> None:
    head = f"{'group':<16}{'SR':>7}{'FTR':>7}{'react p50(ms)':>15}{'react p95(ms)':>15}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        s = r["summary"]
        rs = s.get("reaction_sim_s")
        print(f"{r['label']:<16}{s['success_rate']:>7.3f}"
              f"{s['false_trigger_rate']:>7.3f}"
              f"{fmt_ms(rs, 'p50'):>15}{fmt_ms(rs, 'p95'):>15}")
        if "ledger_ms_mean" in s:
            lg = s["ledger_ms_mean"]
            print(f"  ledger(ms/trial): sense={lg['sense_ms']:.3f} "
                  f"arbiter={lg['arbiter_ms']:.3f} inference={lg['inference_ms']:.3f} "
                  f"exec={lg['exec_ms']:.3f} settle={lg['settle_ms']:.3f} "
                  f"(steps={s['ledger_steps_mean']:.0f}, "
                  f"engagements={s['arbiter_engagements_mean']:.1f})")


def make_figure(rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    # 左：反应延迟剖面直方图（逻辑时间，仅统计有有效响应的 trial）
    bins = np.linspace(0, 300, 31)  # ms
    for r, color in zip(rows, colors):
        lat = [t["reaction_sim_s"] * 1e3 for t in r["profiler"].trials
               if t["reaction_sim_s"] is not None]
        ax1.hist(lat, bins=bins, alpha=0.55, color=color, label=r["label"],
                 edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("reaction latency (ms, sim time)")
    ax1.set_ylabel("trials")
    ax1.set_title("Reaction latency profile (stim -> first valid action)")
    ax1.legend()

    # 右：成功率 + 误触发率柱状图
    x = np.arange(len(rows))
    width = 0.36
    srs = [r["summary"]["success_rate"] for r in rows]
    ftrs = [r["summary"]["false_trigger_rate"] for r in rows]
    bars1 = ax2.bar(x - width / 2, srs, width, color=colors, label="success rate")
    bars2 = ax2.bar(x + width / 2, ftrs, width, color=colors, alpha=0.45,
                    hatch="//", label="false trigger rate")
    for bars in (bars1, bars2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("rate")
    ax2.set_title("Success rate vs false trigger rate")
    ax2.legend(loc="upper center", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M3: 4-group comparison benchmark")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10_000, help="eval seed base")
    ap.add_argument("--lambda0-run", default=str(PROJECT_ROOT / "runs" / "ppo_lambda0"))
    ap.add_argument("--alpha-run", default=str(PROJECT_ROOT / "runs" / "ppo_alpha"))
    ap.add_argument("--fig-out", default=str(PROJECT_ROOT / "assets" / "bench_compare_v2.png"))
    ap.add_argument("--json-out", default=str(PROJECT_ROOT / "assets" / "bench_compare_v2.json"))
    args = ap.parse_args()

    env = PaddleCatchEnv(seed=args.seed)
    baseline = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                                 x_limit=env.x_limit)
    groups = [
        ("baseline", baseline),
        ("PPO lambda=0", load_ppo_policy(args.lambda0_run)),
        ("PPO alpha-dyn", load_ppo_policy(args.alpha_run)),
    ]

    rows = []
    for label, policy in groups:
        print(f"evaluating {label} ...", flush=True)
        prof = evaluate_group(env, policy, args.trials, args.seed, label)
        rows.append({"label": label, "summary": prof.summary(), "profiler": prof})
    env.close()

    print("evaluating dual-path ...", flush=True)
    prof = evaluate_dualpath(args.trials, args.seed, args.alpha_run, "dual-path")
    rows.append({"label": "dual-path", "summary": prof.summary(), "profiler": prof})

    print_table(rows)
    make_figure(rows, Path(args.fig_out))

    payload = [{"label": r["label"], "summary": r["summary"],
                "trials": r["profiler"].trials} for r in rows]
    Path(args.json_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
