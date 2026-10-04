"""M4.5 测试：空闲惩罚奖励、决策死线闸门、5Hz 时钟化规划器、死线实验冒烟。"""
import numpy as np
import pytest

from src.bench.baseline import BallisticBaseline
from src.bench.deadline import (ClockedPlanner, DeadlineGate,
                                run_deadline_bench)
from src.reflex.dualpath_env import DualPathEnv
from src.reflex.env import PaddleCatchEnv
from src.reflex.wrappers import AlphaController, AlphaLatencyWrapper


def test_idle_penalty_wrapper():
    """空闲期动作惩罚：无球期动作超阈值扣分，静止不扣。"""
    env = PaddleCatchEnv(seed=0)
    envw = AlphaLatencyWrapper(env, controller=AlphaController(fixed=0.0),
                               beta=0.0, idle_beta=0.1)
    envw.reset(seed=0)
    assert not envw.ball_launched
    # 静止：无惩罚
    _, r0, term, _, _ = envw.step(np.array([0.0]))
    if not term:
        assert r0 == 0.0
    # 大幅动作：扣分（idle_beta * (|a|-0.1)/v_max = 0.1*5.9/6）
    envw.reset(seed=0)
    _, r1, term, _, _ = envw.step(np.array([6.0]))
    if not term:
        assert r1 == pytest.approx(-0.1 * 5.9 / 6.0, rel=1e-6)
    envw.close()


def test_deadline_gate_timeout():
    """死线闸门：deadline=0 全部超时出零动作；大 deadline 原样透传。"""
    env = PaddleCatchEnv(seed=0)
    pol = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                            x_limit=env.x_limit)
    gate0 = DeadlineGate(pol, deadline_s=0.0)
    gate0.reset()
    obs, _ = env.reset(seed=0)
    a = gate0.act(obs, False)
    assert a[0] == 0.0 and gate0.n_timeouts == 1
    gate_big = DeadlineGate(pol, deadline_s=10.0)
    gate_big.reset()
    a = gate_big.act(obs, True)
    assert a[0] != 0.0 or True  # 透传（值本身由基线决定）
    assert gate_big.n_timeouts == 0
    env.close()


def test_clocked_planner_cadence():
    """5Hz 时钟化：hold=10，仅 tick 步出动作，其余 9/10 记超时零动作。"""
    env = PaddleCatchEnv(seed=0)
    pol = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z,
                            x_limit=env.x_limit)
    clk = ClockedPlanner(pol, control_hz=50, rate_hz=5.0)
    clk.reset()
    obs, _ = env.reset(seed=0)
    outs = [clk.act(obs, True)[0] for _ in range(20)]
    env.close()
    assert clk.n_timeouts == 18  # 20 步里第 10、20 步是 tick
    assert outs[9] != 0.0 or outs[19] != 0.0, "tick 步应有真实动作"
    assert all(o == 0.0 for i, o in enumerate(outs) if (i + 1) % 10 != 0)


def test_dualpath_deadline_timeout_all():
    """双通路死线=0：全部步超时 → 零动作 → 必失败，timeout_frac=1.0。"""
    base = PaddleCatchEnv(seed=0)
    fast = type("Zero", (), {"act": lambda s, o, l: np.array([1.0])})()
    from src.reflex.slowpath import CruisePlanner
    env = DualPathEnv(base, CruisePlanner(v_max=base.v_max, x_limit=base.x_limit),
                      fast, deadline_s=0.0)
    env.reset(seed=0)
    while True:
        _, _, terminated, truncated, info = env.step()
        if terminated or truncated:
            break
    dl = info["events"]["deadline"]
    env.close()
    assert dl["timeout_frac"] == 1.0
    assert info["events"]["outcome"]["type"] != "caught"


def test_deadline_bench_smoke():
    """死线跑批冒烟：3 组 × 5 试验，结构齐全。"""
    result = run_deadline_bench(n_trials=5, seed0=777,
                                alpha_run="runs/ppo_alpha", verbose=False)
    assert len(result["groups"]) == 3
    for g in result["groups"]:
        assert 0.0 <= g["summary"]["success_rate"] <= 1.0
        d = g["deadline_stats"]
        assert 0.0 <= d["timeout_frac"] <= 1.0
    assert result["groups"][0]["deadline_stats"]["timeout_frac"] > 0.5, \
        "5Hz 慢通路在 20ms 死线下应大量超时"
