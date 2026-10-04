"""α 动态延迟奖励（REWARD.md）：AlphaController + AlphaLatencyWrapper。

奖励函数：
    R = 1[接球成功] − α_t · (t_response / t_window) − β·|Δa|/v_max

- t_response = 刺激(stim) → 首次有效动作(first_action) 的逻辑时间差
- t_window   = 刺激 → 球到达接球高度的物理时间窗（由发球参数解析求得）
- α_t 按 REWARD.md 闭环规则逐试验更新（滑动窗成功率 → clamp 更新）
- β 为动作变化率惩罚系数（防抖振，取小值）

与 REWARD.md 超参的差异：α 在每个试验结束后更新（而非每 1000 次评测窗口），
滑动窗仍按窗口均值估计 SR_recent——对偶更新的步进更密、等价平滑，
默认超参（α_min=0, α_max=2.0, SR_target=0.85, η=0.02, N=1000）保持文档原值。
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

GRAVITY = 9.81


def flight_time_to_height(spawn_z: float, vz: float, target_z: float) -> float:
    """解析求球从发球点到达 target_z 高度的最小正时间根（t_window 的物理基准）。"""
    # z(t) = spawn_z + vz*t - g/2*t^2 = target_z
    disc = vz * vz - 2.0 * GRAVITY * (target_z - spawn_z)
    if disc < 0:
        return float("nan")
    sqrt_disc = np.sqrt(disc)
    roots = [(vz + sqrt_disc) / GRAVITY, (vz - sqrt_disc) / GRAVITY]
    positive = [t for t in roots if t > 1e-6]
    return min(positive) if positive else float("nan")


class AlphaController:
    """REWARD.md 第二节：α 动态值（拉格朗日自适应惩罚）。

    α_{t+1} = clamp(α_t + η·(SR_recent − SR_target), α_min, α_max)
    """

    def __init__(self, alpha_min: float = 0.0, alpha_max: float = 2.0,
                 sr_target: float = 0.85, eta: float = 0.02,
                 window: int = 1000, fixed: float | None = None):
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.sr_target = float(sr_target)
        self.eta = float(eta)
        self.fixed = fixed  # 非 None 时冻结 α（λ=0 消融组传 fixed=0.0）
        self.alpha = self.alpha_min if fixed is None else float(fixed)
        self.history: deque[float] = deque(maxlen=int(window))
        self.alpha_log: list[float] = []
        self.n_updates = 0

    def update(self, success: bool) -> float:
        """每个试验结束调用一次，返回更新后的 α。"""
        self.history.append(1.0 if success else 0.0)
        self.n_updates += 1
        if self.fixed is not None:
            self.alpha_log.append(self.alpha)
            return self.alpha
        sr_recent = float(np.mean(self.history))
        self.alpha = float(np.clip(
            self.alpha + self.eta * (sr_recent - self.sr_target),
            self.alpha_min, self.alpha_max))
        self.alpha_log.append(self.alpha)
        return self.alpha

    @property
    def recent_sr(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0


class AlphaLatencyWrapper(gym.Wrapper):
    """把环境的 ±1 终局奖励替换为 REWARD.md 的延迟感知奖励。

    终局步：R = (1 if success else -1) − α·clip(t_response/t_window, 0, max_ratio)
    每步：  R -= β·|a_t − a_{t-1}| / v_max   （防抖振，系数小）
    空闲期（球未发射）：R -= idle_β·max(0, |a|−a_thresh)/v_max
                       （M4.5 奖励工程对照组：试图用奖励压误触发）
    """

    def __init__(self, env: gym.Env, controller: AlphaController | None = None,
                 beta: float = 0.01, max_ratio: float = 2.0,
                 success_outcome: str = "caught", idle_beta: float = 0.0):
        super().__init__(env)
        self.controller = controller or AlphaController()
        self.beta = float(beta)
        self.idle_beta = float(idle_beta)
        self.max_ratio = float(max_ratio)
        # 成功终局名：catch 场景 "caught"，dodge 场景 "dodged"
        self.success_outcome = str(success_outcome)
        self._t_window = 1.0
        self._prev_action = 0.0

    # ---- 供训练回调读取（经 VecEnv.env_method 穿透 Monitor/VecNormalize） ----

    @property
    def ball_launched(self) -> bool:
        return self.env.ball_launched

    def get_alpha(self) -> float:
        return self.controller.alpha

    def get_recent_sr(self) -> float:
        return self.controller.recent_sr

    def get_alpha_log(self) -> list[float]:
        return list(self.controller.alpha_log)

    # ---- Gymnasium 接口 ----

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        spec = info["launch_spec"]
        t_window = flight_time_to_height(
            float(spec.spawn_pos[2]), float(spec.velocity[2]), self.env.catch_z)
        self._t_window = t_window if np.isfinite(t_window) and t_window > 0 else 1.0
        self._prev_action = 0.0
        return obs, info

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(1)[0])
        launched_before = self.env.ball_launched
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 防抖振：动作变化率惩罚（每步）
        rate_penalty = self.beta * abs(a - self._prev_action) / self.env.v_max
        self._prev_action = a

        # 空闲期动作惩罚（奖励工程对照组）：无球期动作超阈值部分扣分
        if self.idle_beta > 0.0 and not launched_before:
            excess = max(0.0, abs(a) - self.env.action_threshold)
            rate_penalty += self.idle_beta * excess / self.env.v_max

        if terminated:
            events = info.get("events", {})
            success = events.get("outcome", {}).get("type") == self.success_outcome
            first_action = events.get("first_action")
            stim = events.get("stim")
            ratio = 0.0
            if first_action and stim:
                ratio = float(np.clip(
                    (first_action["sim"] - stim["sim"]) / self._t_window,
                    0.0, self.max_ratio))
            alpha = self.controller.alpha
            reward = (1.0 if success else -1.0) - alpha * ratio - rate_penalty
            self.controller.update(success)
            info["success"] = bool(success)
            info["alpha"] = alpha
            info["latency_ratio"] = ratio
        else:
            reward -= rate_penalty
        return obs, reward, terminated, truncated, info
