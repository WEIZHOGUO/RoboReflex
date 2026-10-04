"""M3 测试：仲裁器滞回不震荡、驻留时间、非对称握手、mux 排他、误触发率达标。"""
import math

import numpy as np
import pytest

from src.bench.policies import load_ppo_policy
from src.reflex.arbiter import Arbiter, ArbiterConfig, ball_ttc
from src.reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from src.reflex.env import PaddleCatchEnv
from src.reflex.slowpath import CruisePlanner


def test_ball_ttc():
    # 静止悬在 1.5m，接球高度 0.17m：自由落体约 0.52s
    t = ball_ttc([0, 0, 1.5], [0, 0, 0], 0.17)
    assert t == pytest.approx(np.sqrt(2 * (1.5 - 0.17) / 9.81), rel=1e-6)
    # 球已低于接球高度 / 向上飞离 → 无正根
    assert ball_ttc([0, 0, 0.1], [0, 0, -1.0], 0.17) == math.inf


def test_arbiter_hysteresis_no_oscillation():
    """TTC 在滞回带 [enter, exit] 内抖动时状态必须保持不变（不震荡）。"""
    cfg = ArbiterConfig(ttc_enter=0.8, ttc_exit=1.1, dwell_s=0.1)
    arb = Arbiter(cfg)
    # 带内抖动：不应接管
    t = 0.0
    for i in range(100):
        t += 0.02
        dec = arb.decide(0.9 + 0.1 * (i % 2), t)
        assert not dec.engaged
    # 穿越进入阈值：接管一次
    dec = arb.decide(0.5, t + 0.02)
    assert dec.engaged and dec.just_engaged
    # 回到带内抖动：必须保持接管，不乒乓
    t += 0.04
    switches = 0
    for i in range(100):
        t += 0.02
        dec = arb.decide(0.9 + 0.1 * (i % 2), t)
        assert dec.engaged
        switches += dec.just_engaged or dec.just_disengaged
    assert switches == 0
    # 明确越出退出阈值：交还一次
    dec = arb.decide(1.5, t + 0.02)
    assert not dec.engaged and dec.just_disengaged
    assert arb.n_engagements == 1 and arb.n_disengagements == 1


def test_arbiter_dwell_time():
    """最短驻留：切换后 dwell_s 内禁止反向切换。"""
    cfg = ArbiterConfig(ttc_enter=0.8, ttc_exit=1.1, dwell_s=0.1)
    arb = Arbiter(cfg)
    dec = arb.decide(0.5, 1.0)
    assert dec.just_engaged
    # 50ms 后 TTC 已超退出阈值，但驻留未满 → 不允许交还
    dec = arb.decide(2.0, 1.05)
    assert dec.engaged and not dec.just_disengaged
    # 驻留期满后允许交还
    dec = arb.decide(2.0, 1.15)
    assert dec.just_disengaged


def test_arbiter_handover_window():
    """非对称握手：交还过渡窗 window = min(handover_s, k·TTC)。"""
    cfg = ArbiterConfig(ttc_enter=0.8, ttc_exit=1.1, dwell_s=0.0,
                        handover_s=0.02, handover_ttc_k=0.1)
    arb = Arbiter(cfg)
    arb.decide(0.5, 0.0)
    dec = arb.decide(2.0, 1.0)   # k·TTC=0.2 > 20ms → 取 20ms
    assert dec.just_disengaged and dec.handover_window == pytest.approx(0.02)
    arb.reset()
    arb.decide(0.5, 0.0)
    dec = arb.decide(1.15, 1.0)  # k·TTC=0.115 > 20ms → 仍 20ms
    assert dec.handover_window == pytest.approx(0.02)
    cfg2 = ArbiterConfig(ttc_enter=0.8, ttc_exit=1.1, dwell_s=0.0,
                         handover_s=0.02, handover_ttc_k=0.01)
    arb2 = Arbiter(cfg2)
    arb2.decide(0.5, 0.0)
    dec = arb2.decide(1.5, 1.0)  # k·TTC=0.015 < 20ms → 压缩到 15ms
    assert dec.handover_window == pytest.approx(0.015)


def test_arbiter_false_alarm_when_not_launched():
    """红队九.3：无球时处于接管态 = 乱拉警报，记 false alarm 并立即交还。"""
    arb = Arbiter(ArbiterConfig(dwell_s=0.0))
    arb.decide(0.5, 0.0, launched=True)
    assert arb.engaged
    dec = arb.decide(math.inf, 0.05, launched=False)
    assert not dec.engaged and arb.false_alarms == 1


def test_arbiter_multi_on():
    """多物体 O(N)：取 min-TTC 判定。"""
    arb = Arbiter(ArbiterConfig(dwell_s=0.0))
    dec = arb.decide_multi([5.0, math.inf, 0.3], 0.0)
    assert dec.just_engaged


def test_slowpath_low_rate_and_resync():
    """5Hz 规划：指令在规划间隔内保持；resync 后参考轨迹从当前位置续接。"""
    p = CruisePlanner(v_max=6.0, x_limit=0.95, rate_hz=5.0, amplitude=0.25, period=4.0)
    obs = np.zeros(8)
    cmds = [p.act(obs, t)[0] for t in np.arange(0, 1.0, 0.02)]
    # 1 秒 50 步控制里应只规划 ~5 次
    assert 4 <= p.n_plans <= 6
    # 驻留间隔内指令严格不变
    assert len(set(np.round(cmds, 12))) <= p.n_plans
    # resync：把挡板当前位置回放进去，重解相位（不炸、有限值）
    obs[6] = 0.2
    p.resync(obs, 1.0)
    a = p.act(obs, 1.0)
    assert np.isfinite(a[0])


def _make_dual_env(seed=0, **slow_kw):
    base = PaddleCatchEnv(seed=seed)
    slow = CruisePlanner(v_max=base.v_max, x_limit=base.x_limit, **slow_kw)
    fast = load_ppo_policy("runs/ppo_alpha")
    return DualPathEnv(base, slow, fast)


def test_dualpath_mux_exclusive():
    """mux 排他：每步指令来源恰为一路；正常 episode = slow* → fast* 单次切换。"""
    env = _make_dual_env(seed=0)
    for i in range(5):
        run_dualpath_episode(env, seed=200 + i)
        src = env.step_sources
        assert set(src) <= {"slow", "fast", "blend"}
        assert src[0] == "slow", "无球期必须慢通路"
        assert src[-1] == "fast", "接球段必须快通路"
        # 至多一次 slow→fast 切换（blend 只允许出现在 fast→slow 的 20ms 窗内）
        transitions = sum(1 for a, b in zip(src, src[1:]) if a != b)
        assert transitions <= 2, f"乒乓切换: {src}"
        assert env.arbiter.n_engagements == 1
    env.close()


def test_dualpath_false_trigger_and_success():
    """核心验收：双通路误触发率（仲裁器误报）= 0，成功率不塌。"""
    from src.latency.profiler import LatencyProfiler
    env = _make_dual_env(seed=1)
    prof = LatencyProfiler()
    for i in range(20):
        prof.record(run_dualpath_episode(env, seed=300 + i))
    s = prof.summary()
    env.close()
    assert s["false_trigger_rate"] == 0.0
    assert s["success_rate"] >= 0.9
    assert s["reaction_sim_s"]["p95"] <= 0.025
    # 五段式账本齐全且裁判耗时单列
    lg = s["ledger_ms_mean"]
    assert set(lg) == {"sense_ms", "arbiter_ms", "inference_ms", "exec_ms", "settle_ms"}
    assert lg["arbiter_ms"] >= 0.0
    assert s["arbiter_engagements_mean"] == pytest.approx(1.0)
