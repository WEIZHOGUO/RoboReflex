"""ReflexSuite-mini 手写基线：打地鼠 + 滚球拦截。

观测布局与接球环境同构（8 维）：
- 打地鼠：[地鼠x, 0, 地鼠z, 是否探头, 剩余存活, 0, 挡板x, 挡板v]
- 滚球：  [球位置(3), 球速度(3), 挡板x, 挡板v]
接口与 BallisticBaseline 一致：act(obs, active) -> (1,) 目标速度。
"""
from __future__ import annotations

import numpy as np


class MoleBaseline:
    """打地鼠基线：地鼠探头期间直接扑向 mole_x，无目标期纹丝不动。"""

    def __init__(self, v_max: float, x_limit: float, kp: float = 12.0):
        self.v_max = float(v_max)
        self.x_limit = float(x_limit)
        self.kp = float(kp)

    def act(self, obs, active: bool) -> np.ndarray:
        if not active:
            return np.array([0.0])
        obs = np.asarray(obs, dtype=np.float64)
        target = np.clip(obs[0], -self.x_limit, self.x_limit)
        cmd = self.kp * (target - obs[6])
        return np.array([np.clip(cmd, -self.v_max, self.v_max)])


class RollingInterceptBaseline:
    """滚球拦截基线：按当前速度外推球到达挡板线（y=0）时的 x，匀速卡位。

    每步用最新观测重算（滚动摩擦导致的减速被自然吸收）。
    """

    def __init__(self, v_max: float, x_limit: float, kp: float = 12.0):
        self.v_max = float(v_max)
        self.x_limit = float(x_limit)
        self.kp = float(kp)

    def act(self, obs, active: bool) -> np.ndarray:
        if not active:
            return np.array([0.0])
        obs = np.asarray(obs, dtype=np.float64)
        bx, by = float(obs[0]), float(obs[1])
        vx, vy = float(obs[3]), float(obs[4])
        if vy < -1e-6 and by > 0.0:
            t = by / (-vy)
            ix = bx + vx * t
        else:
            ix = bx  # 球已到线/异常：跟踪当前 x
        target = np.clip(ix, -self.x_limit, self.x_limit)
        cmd = self.kp * (target - obs[6])
        return np.array([np.clip(cmd, -self.v_max, self.v_max)])
