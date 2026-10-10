"""ReflexSuite-mini 跑批：三任务（接球/打地鼠/滚球拦截）× 三方案
（手写基线 / PPO α动态 / 机映双通路），各 n 次试验（同一组种子序列）。

每场景每组输出：成功率、反应延迟 p50/p95（逻辑时间）、误触发率、
以及机映特有维度——延迟计入评分：

    latency_factor = min(1, T_control / p50_reaction)   # 越接近步长下限越高
    final_score    = success_rate × latency_factor

产物：
- assets/reflexsuite_bench.json      原始 trial 数据 + 汇总
- assets/reflexsuite_results.md      三任务×三方案完整表（论文/README 直接引用）
- assets/paper/fig_reflexsuite_<task>.png  每场景对比图（延迟剖面+SR/FTR/评分）

用法：
  python -m src.bench.reflexsuite_bench --trials 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from ..reflex.reflexsuite_envs import env_class_for_scenario
from ..reflex.slowpath import CruisePlanner
from .baseline import BallisticBaseline, run_episode
from .policies import load_ppo_policy
from .reflexsuite_baselines import MoleBaseline, RollingInterceptBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUP_LABELS = {"baseline": "baseline", "ppo_alpha": "PPO alpha-dyn",
                "dual_path": "dual-path"}


def _baseline_for(env, task: str):
    if task == "ball_catch":
        return BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                                 x_limit=env.x_limit)
    if task == "whack_a_mole":
        return MoleBaseline(v_max=env.v_max, x_limit=env.x_limit)
    if task == "rolling_ball":
        return RollingInterceptBaseline(v_max=env.v_max, x_limit=env.x_limit)
    raise ValueError(task)


def make_env(task: str, seed: int):
    yaml_path = PROJECT_ROOT / "src" / "scenarios" / f"reflexsuite_{task}.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    return env_class_for_scenario(scenario)(scenario_path=str(yaml_path), seed=seed)


def latency_factor(p50_s: float | None, control_dt: float) -> float:
    """延迟惩罚因子：p50 越接近控制步长下限越接近 1；无有效响应记 0。"""
    if p50_s is None or p50_s <= 0:
        return 0.0
    return float(min(1.0, control_dt / p50_s))


def evaluate_single(env, policy, n_trials: int, seed0: int,
                    label: str) -> LatencyProfiler:
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(run_episode(env, policy, seed=seed0 + i))
        if (i + 1) % 100 == 0 or i + 1 == n_trials:
            print(f"    [{label}] {i + 1}/{n_trials}", flush=True)
    return prof


def evaluate_dualpath(task: str, n_trials: int, seed0: int, run_dir,
                      label: str) -> LatencyProfiler:
    base = make_env(task, seed0)
    slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit)
    fast = load_ppo_policy(run_dir)
    env = DualPathEnv(base, slow, fast)
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(run_dualpath_episode(env, seed=seed0 + i))
        if (i + 1) % 100 == 0 or i + 1 == n_trials:
            print(f"    [{label}] {i + 1}/{n_trials}", flush=True)
    env.close()
    return prof


def summarize(prof: LatencyProfiler, control_dt: float) -> dict:
    s = prof.summary()
    rs = s.get("reaction_sim_s")
    p50 = rs["p50"] if rs else None
    p95 = rs["p95"] if rs else None
    lf = latency_factor(p50, control_dt)
    return {
        "n_trials": s["n_trials"],
        "success_rate": s["success_rate"],
        "false_trigger_rate": s["false_trigger_rate"],
        "reaction_p50_s": p50,
        "reaction_p95_s": p95,
        "latency_factor": lf,
        "final_score": s["success_rate"] * lf,
    }


def make_figure(task: str, rows: list[dict], out_path: Path) -> None:
    labels = [GROUP_LABELS[r["group"]] for r in rows]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    bins = np.linspace(0, 200, 41)  # ms
    for r, color in zip(rows, colors):
        lat = [t["reaction_sim_s"] * 1e3 for t in r["profiler"].trials
               if t["reaction_sim_s"] is not None]
        ax1.hist(lat, bins=bins, alpha=0.55, color=color,
                 label=GROUP_LABELS[r["group"]], edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("reaction latency (ms, sim time)")
    ax1.set_ylabel("trials")
    ax1.set_title("Reaction latency profile (stim -> first valid action)")
    ax1.legend()

    x = np.arange(len(rows))
    width = 0.27
    srs = [r["metrics"]["success_rate"] for r in rows]
    ftrs = [r["metrics"]["false_trigger_rate"] for r in rows]
    scores = [r["metrics"]["final_score"] for r in rows]
    for offs, vals, name, alpha, hatch in [
            (-width, srs, "success rate", 1.0, ""),
            (0.0, ftrs, "false trigger rate", 0.45, "//"),
            (width, scores, "final score (SR x latency factor)", 0.8, "..")]:
        bars = ax2.bar(x + offs, vals, width, color=colors, alpha=alpha,
                       hatch=hatch, label=name)
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.35)
    ax2.set_ylabel("rate / score")
    ax2.set_title("Success / false trigger / latency-penalized score")
    ax2.legend(loc="upper center", fontsize=8)

    fig.suptitle(f"ReflexSuite-mini: {task} (n={rows[0]['metrics']['n_trials']}, "
                 f"50Hz control / 500Hz physics)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  figure saved: {out_path}", flush=True)


def write_results_md(all_rows: dict, out_path: Path) -> None:
    lines = [
        "# ReflexSuite-mini 结果（三任务 × 三方案）",
        "",
        "复刻上交大 ReflexBench 任务精神（接球 / 打地鼠 / 滚球拦截），",
        "机映口径：50Hz 控制 / 500Hz 物理，每格试验数相同、种子序列相同。",
        "",
        "延迟计入评分（机映特有维度，ReflexBench 原基准不含）：",
        "`final_score = success_rate × min(1, T_control / p50_reaction)`，",
        "反应延迟越接近控制步长下限（20ms）因子越接近 1。",
        "",
        "| 任务 | 方案 | SR | FTR | p50 (ms) | p95 (ms) | 延迟因子 | 最终分 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    task_names = {"ball_catch": "接球 Ball Catching",
                  "whack_a_mole": "打地鼠 Whack-a-Mole",
                  "rolling_ball": "滚球拦截 Rolling Ball"}
    for task, rows in all_rows.items():
        for r in rows:
            m = r["metrics"]
            p50 = f"{m['reaction_p50_s'] * 1e3:.1f}" if m["reaction_p50_s"] is not None else "n/a"
            p95 = f"{m['reaction_p95_s'] * 1e3:.1f}" if m["reaction_p95_s"] is not None else "n/a"
            lines.append(
                f"| {task_names[task]} | {GROUP_LABELS[r['group']]} "
                f"| {m['success_rate']:.3f} | {m['false_trigger_rate']:.3f} "
                f"| {p50} | {p95} | {m['latency_factor']:.3f} "
                f"| **{m['final_score']:.3f}** |")
    lines += [
        "",
        "> 诚实声明：本表为对 ReflexBench 任务定义的复刻实现（非官方代码/官方评测），",
        "> 与 ReflexVLA 公开成绩的对照见 README「ReflexSuite-mini」一节。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results md saved: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="ReflexSuite-mini: 3 tasks x 3 systems")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20_000, help="eval seed base")
    ap.add_argument("--json-out",
                    default=str(PROJECT_ROOT / "assets" / "reflexsuite_bench.json"))
    ap.add_argument("--md-out",
                    default=str(PROJECT_ROOT / "assets" / "reflexsuite_results.md"))
    args = ap.parse_args()

    tasks = ["ball_catch", "whack_a_mole", "rolling_ball"]
    run_dirs = {t: PROJECT_ROOT / "runs" / f"ppo_rs_{t}_alpha" for t in tasks}

    all_rows: dict[str, list[dict]] = {}
    for task in tasks:
        print(f"== task: {task} ==", flush=True)
        env = make_env(task, args.seed)
        control_dt = 1.0 / env.control_hz
        rows = []

        print("  evaluating baseline ...", flush=True)
        prof = evaluate_single(env, _baseline_for(env, task),
                               args.trials, args.seed, "baseline")
        rows.append({"group": "baseline",
                     "metrics": summarize(prof, control_dt), "profiler": prof})

        print("  evaluating PPO alpha-dyn ...", flush=True)
        prof = evaluate_single(env, load_ppo_policy(run_dirs[task]),
                               args.trials, args.seed, "ppo_alpha")
        rows.append({"group": "ppo_alpha",
                     "metrics": summarize(prof, control_dt), "profiler": prof})
        env.close()

        print("  evaluating dual-path ...", flush=True)
        prof = evaluate_dualpath(task, args.trials, args.seed,
                                 run_dirs[task], "dual_path")
        rows.append({"group": "dual_path",
                     "metrics": summarize(prof, control_dt), "profiler": prof})

        make_figure(task, rows,
                    PROJECT_ROOT / "assets" / "paper" / f"fig_reflexsuite_{task}.png")
        all_rows[task] = rows

    # 控制台总表
    head = f"{'task':<14}{'group':<14}{'SR':>7}{'FTR':>7}{'p50(ms)':>9}{'p95(ms)':>9}{'lat_fac':>9}{'score':>8}"
    print("\n" + head + "\n" + "-" * len(head))
    for task, rows in all_rows.items():
        for r in rows:
            m = r["metrics"]
            p50 = f"{m['reaction_p50_s'] * 1e3:8.1f}" if m["reaction_p50_s"] is not None else "     n/a"
            p95 = f"{m['reaction_p95_s'] * 1e3:8.1f}" if m["reaction_p95_s"] is not None else "     n/a"
            print(f"{task:<14}{r['group']:<14}{m['success_rate']:>7.3f}"
                  f"{m['false_trigger_rate']:>7.3f}{p50}{p95}"
                  f"{m['latency_factor']:>9.3f}{m['final_score']:>8.3f}")

    write_results_md(all_rows, Path(args.md_out))

    payload = {task: [{"group": r["group"], "metrics": r["metrics"],
                       "trials": r["profiler"].trials} for r in rows]
               for task, rows in all_rows.items()}
    Path(args.json_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
