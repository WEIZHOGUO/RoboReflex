"""双通路环境（机映 RoboReflex M3）：慢规划 × 快反射 × 仲裁器 的排他 mux。

- mux 排他切换：同一时刻只有一路指令到执行器（软过渡窗内例外，见下）
- 非对称握手（红队九.2）：切入反射硬切 0ms；交还慢通路软过渡
  window = min(20ms, k·TTC)，窗内指令线性插值
- 状态交接（四·三.1）：切回瞬间把当前观测回放给慢通路 resync，消除跳变
- 五段式账本（九.1）：传感/裁判/推理/执行/任务完成，墙钟 ns 逐步累计，
  裁判耗时单列；Profiler 消费 events["ledger"]

误触发定义（九.3）：乱拉警报的是仲裁器——本环境把 events["false_trigger"]
覆写为"仲裁器误接管"（无球时接管），环境原 prestim 位移指标保留在
events["prestim_false_trigger"] 供诚实对照（巡航是正常任务行为，非误报）。
"""
from __future__ import annotations

import math
import time

import gymnasium as gym
import numpy as np

from .arbiter import Arbiter, ArbiterConfig, ball_ttc


class DualPathEnv(gym.Wrapper):
    """step(action=None)：忽略外部动作，内部走 传感→裁判→推理(mux)→执行 闭环。"""

    def __init__(self, env: gym.Env, slow_path, fast_path,
                 arbiter: Arbiter | None = None,
                 deadline_s: float | None = None):
        super().__init__(env)
        self.slow_path = slow_path
        self.fast_path = fast_path
        self.arbiter = arbiter or Arbiter(ArbiterConfig())
        # 决策死线（M4.5）：每步 传感+裁判+推理 墙钟耗时超 deadline_s → 该步零动作
        self.deadline_s = deadline_s
        self._obs = None
        self._last_cmd = 0.0
        self._handover_start = -math.inf
        self._handover_window = 0.0
        self._handover_from = 0.0
        self._ledger: dict = {}
        self._decision_ns: list[int] = []
        self._sources: list[str] = []  # 每步指令来源："slow"/"fast"/"blend"（mux 排他验证用）

    @property
    def step_sources(self) -> list[str]:
        return list(self._sources)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        self._last_cmd = 0.0
        self.arbiter.reset()
        self.slow_path.reset(sim_time=0.0)
        self._handover_start = -math.inf
        self._handover_window = 0.0
        self._ledger = {"sense_ns": 0, "arbiter_ns": 0, "inference_ns": 0,
                        "exec_ns": 0, "settle_ns": 0, "n_steps": 0,
                        "timeout_steps": 0}
        self._decision_ns = []
        self._sources = []
        return obs, info

    def step(self, action=None):
        base = self.env.unwrapped
        led = self._ledger

        # ---- 传感段：观测就绪 + TTC 计算（几何量，微秒级） ----
        t0 = time.perf_counter_ns()
        obs = self._obs
        sim_time = float(base.data.time)
        launched = base.ball_launched
        ttc = ball_ttc(obs[0:3], obs[3:6], base.catch_z) if launched else math.inf
        t1 = time.perf_counter_ns()
        led["sense_ns"] += t1 - t0

        # ---- 裁判段：滞回 + 驻留仲裁（单列耗时） ----
        dec = self.arbiter.decide(ttc, sim_time, launched)
        t2 = time.perf_counter_ns()
        led["arbiter_ns"] += t2 - t1

        if dec.just_disengaged:
            # 状态交接：观测回放给慢通路重同步，然后进入软过渡窗
            self.slow_path.resync(obs, sim_time)
            self._handover_start = sim_time
            self._handover_window = dec.handover_window
            self._handover_from = self._last_cmd

        # ---- 推理段：mux 排他选择一路 ----
        if dec.engaged:
            cmd = float(self.fast_path.act(obs, launched)[0])
            src = "fast"
        else:
            cmd = float(self.slow_path.act(obs, sim_time)[0])
            src = "slow"
            w_elapsed = sim_time - self._handover_start
            if 0.0 <= w_elapsed < self._handover_window and self._handover_window > 0:
                w = w_elapsed / self._handover_window
                cmd = (1.0 - w) * self._handover_from + w * cmd
                src = "blend"
        t3 = time.perf_counter_ns()
        led["inference_ns"] += t3 - t2

        # ---- 决策死线：传感+裁判+推理 总墙钟超 deadline → 该步零动作 ----
        decision_ns = t3 - t0
        self._decision_ns.append(decision_ns)
        if self.deadline_s is not None and decision_ns / 1e9 > self.deadline_s:
            cmd = 0.0
            src = src + "|timeout"
            led["timeout_steps"] += 1

        # ---- 执行段：物理步进 ----
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([cmd], dtype=np.float64))
        t4 = time.perf_counter_ns()
        led["exec_ns"] += t4 - t3

        # ---- 任务完成段：终局记账 ----
        self._obs = obs
        self._last_cmd = cmd
        self._sources.append(src)
        led["n_steps"] += 1
        if terminated or truncated:
            events = info.get("events")
            if events is not None:
                events["prestim_false_trigger"] = events.get("false_trigger", False)
                # 误触发重定义（九.3）：仲裁器乱拉警报才算误报
                events["false_trigger"] = self.arbiter.false_alarms > 0
                events["arbiter"] = {
                    "engagements": self.arbiter.n_engagements,
                    "disengagements": self.arbiter.n_disengagements,
                    "false_alarms": self.arbiter.false_alarms,
                }
                events["ledger"] = {
                    "sense_ms": led["sense_ns"] / 1e6,
                    "arbiter_ms": led["arbiter_ns"] / 1e6,
                    "inference_ms": led["inference_ns"] / 1e6,
                    "exec_ms": led["exec_ns"] / 1e6,
                    "settle_ms": led["settle_ns"] / 1e6,
                    "n_steps": led["n_steps"],
                }
                if self.deadline_s is not None and self._decision_ns:
                    arr = np.asarray(self._decision_ns)
                    events["deadline"] = {
                        "deadline_s": self.deadline_s,
                        "timeout_steps": led["timeout_steps"],
                        "timeout_frac": led["timeout_steps"] / max(led["n_steps"], 1),
                        "decision_ms_p50": float(np.percentile(arr, 50) / 1e6),
                        "decision_ms_p99": float(np.percentile(arr, 99) / 1e6),
                    }
                events["switch_sequence"] = self._sources
        led["settle_ns"] += time.perf_counter_ns() - t4
        return obs, reward, terminated, truncated, info


def run_dualpath_episode(env: DualPathEnv, seed: int | None = None) -> dict:
    """跑一次双通路完整试验，返回事件字典（含 ledger / arbiter 统计）。"""
    env.reset(seed=seed)
    while True:
        _, _, terminated, truncated, info = env.step()
        if terminated or truncated:
            return info.get("events", {"outcome": {"type": "truncated"}})
