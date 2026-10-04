"""仲裁器（机映 RoboReflex M3）：几何阈值危险判定 + 防震荡设计。

设计来源 ARCHITECTURE.md 四·三.1 / 八.1 / 九.1/九.7：
- 危险判定：球已发射 且 min-TTC < 阈值（TTC=球到达接球高度的解析时间，微秒级）
- 滞回比较器：TTC 进入阈值 < 退出阈值（危险信号越近越危险，进入比退出更"近"）
- 最短驻留时间（dwell）：每次切换后 dwell_s 内不允许反向切换，杜绝乒乓
- 非对称握手：切入反射=硬切立即生效；交还慢通路=软过渡窗
  window = min(handover_s, k·TTC)（红队九.2，TTC 越小过渡窗越被压缩）
- O(N)：多物体时取 min-TTC（物体×保护区线性扫描），非 O(N^2)

裁判只做几何判定，不含学习组件；AI 只在离线慢环优化其参数（FAQ 八.1）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .wrappers import flight_time_to_height


def ball_ttc(ball_pos, ball_vel, catch_z: float) -> float:
    """球当前状态到接球高度的 TTC（秒）；不会到达（在下方/正远离）返回 inf。"""
    bz, vz = float(ball_pos[2]), float(ball_vel[2])
    if bz <= catch_z:
        return math.inf
    t = flight_time_to_height(bz, vz, catch_z)
    return t if np.isfinite(t) else math.inf


@dataclass
class ArbiterConfig:
    ttc_enter: float = 0.8      # TTC 低于此值 → 反射接管（需大于场景最大发球飞行时间）
    ttc_exit: float = 1.1       # TTC 高于此值 → 交还慢通路（滞回带 [enter, exit]）
    dwell_s: float = 0.1        # 最短驻留时间（每次切换后 100ms 内禁止反向切换）
    handover_s: float = 0.02    # 交还慢通路的软过渡窗上限（20ms）
    handover_ttc_k: float = 0.1 # window = min(handover_s, k·TTC) 的 k


@dataclass
class ArbiterDecision:
    engaged: bool               # 本步是否由反射通路接管
    just_engaged: bool = False  # 本步发生硬切（慢→快）
    just_disengaged: bool = False  # 本步开始交还（快→慢，进入软过渡窗）
    handover_window: float = 0.0   # 交还过渡窗时长（秒），仅 just_disengaged 时有意义


@dataclass
class Arbiter:
    """滞回 + 驻留仲裁器。decide() 纯函数式，微秒级。"""
    cfg: ArbiterConfig = field(default_factory=ArbiterConfig)

    def __post_init__(self):
        assert self.cfg.ttc_enter < self.cfg.ttc_exit, "滞回带要求 enter < exit（TTC 语义）"
        self.engaged = False
        self.last_switch_t = -math.inf
        self.n_engagements = 0
        self.n_disengagements = 0
        self.false_alarms = 0  # 无威胁时接管（乱拉警报）次数

    def reset(self):
        self.engaged = False
        self.last_switch_t = -math.inf
        self.n_engagements = 0
        self.n_disengagements = 0
        self.false_alarms = 0

    def decide(self, ttc: float, sim_time: float, launched: bool = True) -> ArbiterDecision:
        """单次判定。launched=False 时任何接管都记 false alarm（红队九.3）。"""
        cfg = self.cfg
        dwell_ok = (sim_time - self.last_switch_t) >= cfg.dwell_s
        dec = ArbiterDecision(engaged=self.engaged)

        if not self.engaged:
            if launched and ttc < cfg.ttc_enter and dwell_ok:
                # 切入反射：硬切，立即生效
                self.engaged = True
                self.last_switch_t = sim_time
                self.n_engagements += 1
                dec.engaged = True
                dec.just_engaged = True
        else:
            if not launched:
                # 球未发射却处于接管态 = 误拉警报，立即交还
                self.false_alarms += 1
                self.engaged = False
                self.last_switch_t = sim_time
                self.n_disengagements += 1
                dec.engaged = False
                dec.just_disengaged = True
                dec.handover_window = cfg.handover_s
            elif ttc > cfg.ttc_exit and dwell_ok:
                # 交还慢通路：软过渡窗 window = min(handover_s, k·TTC)
                self.engaged = False
                self.last_switch_t = sim_time
                self.n_disengagements += 1
                dec.engaged = False
                dec.just_disengaged = True
                dec.handover_window = min(cfg.handover_s, cfg.handover_ttc_k * ttc)
        return dec

    def decide_multi(self, ttcs, sim_time: float, launched: bool = True) -> ArbiterDecision:
        """多物体：O(N) 取 min-TTC（九.7），再按单物体逻辑判定。"""
        finite = [t for t in ttcs if math.isfinite(t)]
        return self.decide(min(finite) if finite else math.inf, sim_time, launched)
