"""慢速规划通路（机映 RoboReflex M3）：低频巡航规划器，代表"正常任务"。

- 5Hz 规划频率：两次规划之间保持上一条指令（模拟慢环）
- 无威胁时绕中线正弦巡航（x_ref = A·sin(2πt/T)），可配定点驻留（A=0）
- 状态交接（ARCHITECTURE 四·三.1）：切回时 resync() 把当前观测回放进来，
  重解巡航相位使 x_ref(t0) = 当前挡板位置，消除控制跳变

接口与 BallisticBaseline/PPOPolicy 一致：act(obs, ...) -> (1,) 目标速度。
"""
from __future__ import annotations

import numpy as np


class CruisePlanner:
    """低频正弦巡航规划器。act(obs, sim_time) -> 挡板目标速度（m/s）。"""

    def __init__(self, v_max: float, x_limit: float, rate_hz: float = 5.0,
                 amplitude: float = 0.25, period: float = 4.0, kp: float = 6.0,
                 cruise_v_max: float | None = None):
        self.v_max = float(v_max)
        self.x_limit = float(x_limit)
        self.rate_hz = float(rate_hz)
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.kp = float(kp)
        # 巡航限速（默认半速，巡航本就该"悠闲"）
        self.cruise_v_max = float(cruise_v_max) if cruise_v_max else 0.5 * self.v_max
        self._hold_dt = 1.0 / self.rate_hz
        self._phase = 0.0        # 巡航相位偏移（resync 时重解）
        self._last_plan_t = -np.inf
        self._cmd = 0.0
        self.n_plans = 0         # 统计：实际规划次数（验证低频）

    def reset(self, sim_time: float = 0.0):
        self._phase = 0.0
        self._last_plan_t = -np.inf
        self._cmd = 0.0
        self.n_plans = 0

    def resync(self, obs, sim_time: float):
        """状态交接：把当前观测回放进来，重解相位使参考轨迹从当前位置平滑续接。"""
        x = float(np.asarray(obs)[6])
        if self.amplitude > 1e-9:
            s = np.clip(x / self.amplitude, -1.0, 1.0)
            omega = 2.0 * np.pi / self.period
            self._phase = float(np.arcsin(s) - omega * sim_time)
        self._last_plan_t = -np.inf  # 下一步强制重新规划

    def _ref(self, sim_time: float) -> tuple[float, float]:
        omega = 2.0 * np.pi / self.period
        th = omega * sim_time + self._phase
        x_ref = self.amplitude * np.sin(th)
        v_ref = self.amplitude * omega * np.cos(th)
        return float(np.clip(x_ref, -self.x_limit, self.x_limit)), float(v_ref)

    def act(self, obs, sim_time: float) -> np.ndarray:
        if sim_time - self._last_plan_t >= self._hold_dt - 1e-9:
            x_ref, v_ref = self._ref(sim_time)
            x = float(np.asarray(obs)[6])
            cmd = self.kp * (x_ref - x) + v_ref
            self._cmd = float(np.clip(cmd, -self.cruise_v_max, self.cruise_v_max))
            self._last_plan_t = sim_time
            self.n_plans += 1
        return np.array([self._cmd])
