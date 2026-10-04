"""手写基线：弹道解析预测落点 → 挡板匀速（限速）移向拦截点。

观测布局见 PaddleCatchEnv：[球位置(3), 球速度(3), 挡板位置(1), 挡板速度(1)]。
发球前输出零速度（避免无球期误触发），发球后按解析解拦截。
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81


class BallisticBaseline:
    """弹道解析拦截控制器。act(obs, launched) -> 挡板目标速度（m/s）。"""

    def __init__(self, v_max: float, catch_z: float, x_limit: float, kp: float = 12.0):
        self.v_max = float(v_max)
        self.catch_z = float(catch_z)
        self.x_limit = float(x_limit)
        self.kp = float(kp)

    def predict_intercept_x(self, ball_pos, ball_vel) -> float:
        """解 z(t)=catch_z 的最小正根，外推 x 落点；无解则退化为跟踪当前球 x。"""
        bz, vz = float(ball_pos[2]), float(ball_vel[2])
        # z(t) = bz + vz*t - g/2*t^2 = catch_z
        # => g/2*t^2 - vz*t + (catch_z - bz) = 0
        disc = vz * vz - 2.0 * GRAVITY * (self.catch_z - bz)
        if disc < 0:
            return float(ball_pos[0])
        sqrt_disc = np.sqrt(disc)
        roots = [(vz + sqrt_disc) / GRAVITY, (vz - sqrt_disc) / GRAVITY]
        positive = [t for t in roots if t > 1e-6]
        t = min(positive) if positive else 0.0
        return float(ball_pos[0] + ball_vel[0] * t)

    def act(self, obs, launched: bool) -> np.ndarray:
        if not launched:
            return np.array([0.0])
        obs = np.asarray(obs, dtype=np.float64)
        target = np.clip(self.predict_intercept_x(obs[0:3], obs[3:6]),
                         -self.x_limit, self.x_limit)
        cmd = self.kp * (target - obs[6])
        return np.array([np.clip(cmd, -self.v_max, self.v_max)])


def run_episode(env, policy, seed: int | None = None) -> dict:
    """跑一次完整试验，返回环境给出的事件字典。"""
    obs, _ = env.reset(seed=seed)
    while True:
        action = policy.act(obs, env.ball_launched)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info.get("events", {"outcome": {"type": "truncated"}})


class DodgeBaseline:
    """避障手写基线：解析预测落点，落点在威胁半径内则向远离侧撤离，否则不动。

    与接球基线同一弹道解（predict_intercept_x），动作语义相反。
    act(obs, launched) -> 挡板目标速度（m/s）。
    """

    def __init__(self, v_max: float, catch_z: float, x_limit: float,
                 dodge_radius: float = 0.22, margin: float = 0.08, kp: float = 12.0):
        self.v_max = float(v_max)
        self.x_limit = float(x_limit)
        self.dodge_radius = float(dodge_radius)
        self.margin = float(margin)
        self.kp = float(kp)
        self._catch = BallisticBaseline(v_max, catch_z, x_limit)

    def act(self, obs, launched: bool) -> np.ndarray:
        if not launched:
            return np.array([0.0])
        obs = np.asarray(obs, dtype=np.float64)
        ix = self._catch.predict_intercept_x(obs[0:3], obs[3:6])
        px = float(obs[6])
        if abs(ix - px) >= self.dodge_radius:
            return np.array([0.0])  # 弹道不经过挡板：纹丝不动（不误触发）
        # 落点在威胁半径内：把挡板中心撤到落点的远离侧安全距离
        direction = 1.0 if px >= ix else -1.0
        target = np.clip(ix + direction * (self.dodge_radius + self.margin),
                         -self.x_limit, self.x_limit)
        cmd = self.kp * (target - px)
        return np.array([np.clip(cmd, -self.v_max, self.v_max)])
