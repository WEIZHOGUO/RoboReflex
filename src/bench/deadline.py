"""决策死线实验（M4.5，审稿人反击②）：硬死线 20ms 下谁还能干活。

规则：每步决策（传感+裁判+推理）墙钟耗时 > 20ms（1 个控制周期）→ 该步零动作。
三组：
- slow 5Hz + deadline：手写弹道规划器按 5Hz 节奏出决策（200ms 才出一次），
  非 tick 步视为死线超时 → 零动作。代表"慢速规划通路单独上"。
- PPO alpha + deadline：单通路 PPO，逐步实测墙钟决策时间过闸。
- dual-path + deadline：机映双通路，传感+裁判+推理全部计入决策时间。

预期叙事：死线压缩时慢规划大量超时、成功率崩塌；双通路微秒级裁判+
亚毫秒推理，成功率不掉——"慢规划不拖累快反射"。

用法：python -m src.bench.deadline --trials 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..latency.profiler import LatencyProfiler
from ..reflex.dualpath_env import DualPathEnv
from ..reflex.env import PaddleCatchEnv
from ..reflex.slowpath import CruisePlanner
from .baseline import BallisticBaseline
from .policies import load_ppo_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeadlineGate:
    """墙钟死线闸门：policy.act 实测耗时超 deadline → 该步零动作。"""

    def __init__(self, policy, deadline_s: float):
        self.policy = policy
        self.deadline_s = float(deadline_s)
        self.n_steps = 0
        self.n_timeouts = 0
        self.decision_ns: list[int] = []

    def reset(self):
        self.n_steps = 0
        self.n_timeouts = 0
        self.decision_ns = []

    def act(self, obs, launched: bool) -> np.ndarray:
        t0 = time.perf_counter_ns()
        a = self.policy.act(obs, launched)
        dt = time.perf_counter_ns() - t0
        self.n_steps += 1
        self.decision_ns.append(dt)
        if dt / 1e9 > self.deadline_s:
            self.n_timeouts += 1
            return np.zeros(1)
        return a

    def stats(self) -> dict:
        arr = np.asarray(self.decision_ns or [0])
        return {
            "n_steps": self.n_steps,
            "timeout_steps": self.n_timeouts,
            "timeout_frac": self.n_timeouts / max(self.n_steps, 1),
            "decision_ms_p50": float(np.percentile(arr, 50) / 1e6),
            "decision_ms_p99": float(np.percentile(arr, 99) / 1e6),
        }


class ClockedPlanner:
    """低频规划节奏 + 死线语义：只有规划 tick 步出动作，其余步超时=零动作。

    5Hz 规划器在 50Hz 控制下每 10 步才有一次新决策；硬死线不允许
    沿用陈旧指令（超时=零动作），所以非 tick 步全部输出 0。
    """

    def __init__(self, policy, control_hz: int = 50, rate_hz: float = 5.0,
                 deadline_s: float = 0.02):
        self.policy = policy
        self.hold = max(1, round(control_hz / rate_hz))
        self.deadline_s = float(deadline_s)
        self.n_steps = 0
        self.n_timeouts = 0
        self.decision_ns: list[int] = []
        self._i = 0

    def reset(self):
        self.n_steps = 0
        self.n_timeouts = 0
        self.decision_ns = []
        self._i = 0

    def act(self, obs, launched: bool) -> np.ndarray:
        self._i += 1
        self.n_steps += 1
        if self._i % self.hold != 0:
            # 非规划 tick：5Hz 通路此步无新决策 → 死线超时
            self.n_timeouts += 1
            self.decision_ns.append(200_000_000)  # 决策周期 200ms（逻辑量）
            return np.zeros(1)
        t0 = time.perf_counter_ns()
        a = self.policy.act(obs, launched)
        dt = time.perf_counter_ns() - t0
        self.decision_ns.append(dt)
        if dt / 1e9 > self.deadline_s:
            self.n_timeouts += 1
            return np.zeros(1)
        return a

    def stats(self) -> dict:
        arr = np.asarray(self.decision_ns or [0])
        return {
            "n_steps": self.n_steps,
            "timeout_steps": self.n_timeouts,
            "timeout_frac": self.n_timeouts / max(self.n_steps, 1),
            "decision_ms_p50": float(np.percentile(arr, 50) / 1e6),
            "decision_ms_p99": float(np.percentile(arr, 99) / 1e6),
        }


def _run_episode(env, policy, seed: int) -> dict:
    policy.reset()
    obs, _ = env.reset(seed=seed)
    while True:
        action = policy.act(obs, env.ball_launched)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info.get("events", {"outcome": {"type": "truncated"}})


def run_deadline_bench(n_trials: int, seed0: int, alpha_run,
                       deadline_s: float = 0.02, verbose: bool = True) -> dict:
    rows = []

    # 组 1：慢速规划通路单独（5Hz 节奏 + 死线）
    env = PaddleCatchEnv(seed=seed0)
    baseline = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                                 x_limit=env.x_limit)
    clocked = ClockedPlanner(baseline, control_hz=env.control_hz,
                             rate_hz=5.0, deadline_s=deadline_s)
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(_run_episode(env, clocked, seed0 + i))
    rows.append({"label": "slow-5Hz", "summary": prof.summary(),
                 "deadline_stats": clocked.stats()})
    env.close()

    # 组 2：PPO α 单通路 + 死线闸门（墙钟实测）
    env = PaddleCatchEnv(seed=seed0)
    gate = DeadlineGate(load_ppo_policy(alpha_run), deadline_s)
    prof = LatencyProfiler()
    for i in range(n_trials):
        prof.record(_run_episode(env, gate, seed0 + i))
    rows.append({"label": "PPO alpha", "summary": prof.summary(),
                 "deadline_stats": gate.stats()})
    env.close()

    # 组 3：双通路（传感+裁判+推理全计入决策时间）
    base = PaddleCatchEnv(seed=seed0)
    env = DualPathEnv(base, CruisePlanner(v_max=base.v_max, x_limit=base.x_limit),
                      load_ppo_policy(alpha_run), deadline_s=deadline_s)
    prof = LatencyProfiler()
    decision_ns_all: list[int] = []
    timeout_steps = 0
    for i in range(n_trials):
        env.reset(seed=seed0 + i)
        while True:
            _, _, terminated, truncated, info = env.step()
            if terminated or truncated:
                break
        events = info.get("events", {})
        prof.record(events)
        dl = events.get("deadline", {})
        timeout_steps += dl.get("timeout_steps", 0)
        decision_ns_all.extend(env._decision_ns)
    arr = np.asarray(decision_ns_all or [0])
    rows.append({"label": "dual-path", "summary": prof.summary(),
                 "deadline_stats": {
                     "n_steps": int(len(decision_ns_all)),
                     "timeout_steps": int(timeout_steps),
                     "timeout_frac": timeout_steps / max(len(decision_ns_all), 1),
                     "decision_ms_p50": float(np.percentile(arr, 50) / 1e6),
                     "decision_ms_p99": float(np.percentile(arr, 99) / 1e6),
                 }})
    env.close()

    if verbose:
        head = (f"{'group':<12}{'SR':>7}{'FTR':>7}{'p95(ms)':>9}"
                f"{'timeout%':>10}{'decide p99(ms)':>15}")
        print("\n" + head)
        print("-" * len(head))
        for r in rows:
            s, d = r["summary"], r["deadline_stats"]
            rs = s.get("reaction_sim_s") or {}
            print(f"{r['label']:<12}{s['success_rate']:>7.3f}"
                  f"{s['false_trigger_rate']:>7.3f}"
                  f"{rs.get('p95', float('nan')) * 1e3:>9.1f}"
                  f"{d['timeout_frac'] * 100:>9.1f}%"
                  f"{d['decision_ms_p99']:>15.3f}")
    return {"deadline_s": deadline_s, "n_trials": n_trials, "groups": rows}


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M4.5: decision deadline benchmark")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=40_000)
    ap.add_argument("--alpha-run", default=str(PROJECT_ROOT / "runs" / "ppo_alpha"))
    ap.add_argument("--deadline-ms", type=float, default=20.0)
    ap.add_argument("--json-out",
                    default=str(PROJECT_ROOT / "assets" / "deadline.json"))
    args = ap.parse_args()

    result = run_deadline_bench(args.trials, args.seed, args.alpha_run,
                                deadline_s=args.deadline_ms / 1e3)
    Path(args.json_out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"json saved: {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
