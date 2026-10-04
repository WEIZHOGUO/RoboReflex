"""论文图表一键脚本（M4）：读 assets/*.json + runs/ppo_alpha/train_meta.json，
生成全套论文图到 assets/paper/（每张 PNG + PDF）。

依赖数据（缺哪张跳过哪张，不炸）：
- ablation.json        -> fig_ablation_pareto   消融帕累托散点（B/C×3/D）
- multiseed.json       -> fig_multiseed         多种子误差棒（双通路 × 5 种子）
- dodge.json           -> fig_scenarios         双场景对比（接球 vs 避障）
- speed_sweep.json     -> fig_speed             成功率/延迟 vs 球速曲线
- bench_compare_v2.json-> fig_latency_profile   四组延迟剖面
- runs/ppo_alpha/train_meta.json -> fig_alpha_evolution  α 演化曲线

用法：python -m src.bench.make_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT_ROOT / "assets"


def _load(path: Path):
    if not path.exists():
        print(f"skip: {path.name} not found", flush=True)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"figure saved: {out_dir / name}.png/.pdf", flush=True)


def fig_ablation_pareto(data, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"B": "#DD8452", "C": "#937860", "D": "#55A868"}
    # 同坐标点的标注纵向错开，防止叠字
    seen: dict[tuple, int] = {}
    for row in data:
        s = row["summary"]
        p95 = round(s["reaction_sim_s"]["p95"] * 1e3, 1)
        sr = round(s["success_rate"] * 100, 1)
        group = row["label"][0]
        ax.scatter(p95, sr, s=90, color=colors.get(group, "#4C72B0"), zorder=3)
        dup = seen.get((p95, sr), 0)
        seen[(p95, sr)] = dup + 1
        offset = (8, 6 + 12 * dup) if dup else (8, -14)
        ax.annotate(row["label"], (p95, sr), textcoords="offset points",
                    xytext=offset, fontsize=9)
    ax.set_xlabel("reaction latency p95 (ms, sim time)  -> faster")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Ablation: fixed lambda points vs alpha-dynamic (Pareto)")
    ax.set_ylim(80, 105)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    _save(fig, out_dir, "fig_ablation_pareto")


def fig_multiseed(data, out_dir):
    agg = data["aggregate"]
    seeds = [str(s) for s in data["seeds"]]
    panels = [("success_rate", "success rate", (0, 1.15)),
              ("false_trigger_rate", "false trigger rate", (0, 1.15)),
              ("react_p95_ms", "reaction p95 (ms)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title, ylim) in zip(axes, panels):
        vals = agg[key]["values"]
        ax.bar([0], [agg[key]["mean"]], yerr=[agg[key]["std"]], capsize=6,
               color="#C44E52", alpha=0.8, width=0.4)
        ax.scatter(np.zeros(len(vals)), vals, color="black", s=14, zorder=3)
        if agg[key]["std"] > 0:
            for v, s in zip(vals, seeds):
                ax.annotate(s, (0, v), textcoords="offset points",
                            xytext=(8, -3), fontsize=7, color="gray")
        else:
            ax.annotate("all seeds identical", (0, agg[key]["mean"]),
                        textcoords="offset points", xytext=(8, -3),
                        fontsize=8, color="gray")
        ax.set_xticks([0], ["dual-path\n(5 seeds)"])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Multi-seed robustness (mean +/- std)")
    fig.tight_layout()
    _save(fig, out_dir, "fig_multiseed")


def fig_scenarios(dodge, v2, out_dir):
    # 接球场景取 v2 的 baseline / dual-path；避障场景取 dodge.json 两组
    catch = {r["label"]: r["summary"] for r in v2}
    labels = ["catch\nbaseline", "catch\ndual-path",
              "dodge\nbaseline", "dodge\ndual-path"]
    srs = [catch["baseline"]["success_rate"], catch["dual-path"]["success_rate"],
           dodge["groups"][0]["summary"]["success_rate"],
           dodge["groups"][1]["summary"]["success_rate"]]
    ftrs = [catch["baseline"]["false_trigger_rate"],
            catch["dual-path"]["false_trigger_rate"],
            dodge["groups"][0]["summary"]["false_trigger_rate"],
            dodge["groups"][1]["summary"]["false_trigger_rate"]]
    colors = ["#4C72B0", "#C44E52", "#4C72B0", "#C44E52"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(4)
    for ax, vals, title in ((ax1, srs, "success rate"),
                            (ax2, ftrs, "false trigger rate (arbiter false alarm)")):
        bars = ax.bar(x, vals, 0.6, color=colors)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    ho = dodge["handover"]
    fig.suptitle(f"Second scenario (dodge): soft handover in "
                 f"{ho['trials_with_disengagement']} trials, "
                 f"{ho['blend_steps_total']} blend steps")
    fig.tight_layout()
    _save(fig, out_dir, "fig_scenarios")


def fig_speed(data, out_dir):
    speeds = [r["speed"] for r in data["rows"]]
    b_sr = [r["baseline"]["success_rate"] for r in data["rows"]]
    d_sr = [r["dual_path"]["success_rate"] for r in data["rows"]]
    b_p95 = [r["baseline"]["reaction_sim_s"]["p95"] * 1e3 for r in data["rows"]]
    d_p95 = [r["dual_path"]["reaction_sim_s"]["p95"] * 1e3 for r in data["rows"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(speeds, b_sr, "o-", color="#4C72B0", label="baseline")
    ax1.plot(speeds, d_sr, "s-", color="#C44E52", label="dual-path")
    ax1.set_xlabel("ball speed (m/s)")
    ax1.set_ylabel("success rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Success rate vs ball speed")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(speeds, b_p95, "o-", color="#4C72B0", label="baseline")
    ax2.plot(speeds, d_p95, "s-", color="#C44E52", label="dual-path")
    ax2.set_xlabel("ball speed (m/s)")
    ax2.set_ylabel("reaction latency p95 (ms)")
    ax2.set_title("Latency p95 vs ball speed")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig_speed")


def fig_latency_profile(v2, out_dir):
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 300, 31)
    for r, c in zip(v2, colors):
        lat = [t["reaction_sim_s"] * 1e3 for t in r["trials"]
               if t.get("reaction_sim_s") is not None]
        ax.hist(lat, bins=bins, alpha=0.55, color=c, label=r["label"],
                edgecolor="white", linewidth=0.3)
    ax.set_xlabel("reaction latency (ms, sim time)")
    ax.set_ylabel("trials")
    ax.set_title("Reaction latency profile (stim -> first valid action)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig_latency_profile")


def fig_alpha_evolution(meta, out_dir):
    logs = meta["alpha_log_per_env"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, log in enumerate(logs):
        ax.plot(log, alpha=0.7, label=f"env {i}")
    ax.axhline(2.0, ls="--", color="gray", lw=1, label="alpha_max")
    ax.axhline(0.0, ls=":", color="gray", lw=1)
    ax.set_xlabel("training episode")
    ax.set_ylabel("alpha (lagrange multiplier estimate)")
    ax.set_title("Alpha evolution under closed-loop update "
                 "(SR_target=0.85, eta=0.02)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig_alpha_evolution")


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4: one-shot paper figures")
    ap.add_argument("--out", default=str(ASSETS / "paper"))
    args = ap.parse_args()
    out_dir = Path(args.out)

    data = _load(ASSETS / "ablation.json")
    if data:
        fig_ablation_pareto(data, out_dir)
    data = _load(ASSETS / "multiseed.json")
    if data:
        fig_multiseed(data, out_dir)
    dodge = _load(ASSETS / "dodge.json")
    v2 = _load(ASSETS / "bench_compare_v2.json")
    if dodge and v2:
        fig_scenarios(dodge, v2, out_dir)
    data = _load(ASSETS / "speed_sweep.json")
    if data:
        fig_speed(data, out_dir)
    if v2:
        fig_latency_profile(v2, out_dir)
    meta = _load(PROJECT_ROOT / "runs" / "ppo_alpha" / "train_meta.json")
    if meta and meta.get("alpha_log_per_env"):
        fig_alpha_evolution(meta, out_dir)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
