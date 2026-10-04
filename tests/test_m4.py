"""M4 测试：避障场景、固定 λ 训练模式、软过渡真实触发、速度扫描冒烟。"""
import numpy as np
import pytest

from src.bench.baseline import DodgeBaseline, run_episode
from src.bench.dodge_bench import run_dodge_bench
from src.bench.speed_sweep import run_sweep
from src.latency.profiler import LatencyProfiler
from src.reflex.dualpath_env import DualPathEnv, run_dualpath_episode
from src.reflex.env import PaddleCatchEnv
from src.reflex.slowpath import CruisePlanner
from src.reflex.train import build_vec_env
from src.bench.policies import load_ppo_policy

DODGE_YAML = "src/scenarios/dodge_easy.yaml"


def test_dodge_env_outcomes():
    """dodge 模式：静止时威胁球=hit（失败），偏球=dodged（成功）。"""
    env = PaddleCatchEnv(scenario_path=DODGE_YAML, seed=0)
    assert env.mode == "dodge"
    stay = type("Stay", (), {"act": lambda s, o, l: np.array([0.0])})()
    outcomes = [run_episode(env, stay, seed=i)["outcome"]["type"] for i in range(20)]
    env.close()
    assert set(outcomes) <= {"hit", "dodged"}
    assert "hit" in outcomes, "窄角发球下应有真实威胁球"
    assert "dodged" in outcomes, "应有自然偏出的球"


def test_dodge_baseline_works():
    """手写避障基线：成功率高且零误触发（不威胁就不动）。"""
    env = PaddleCatchEnv(scenario_path=DODGE_YAML, seed=0)
    pol = DodgeBaseline(v_max=env.v_max, catch_z=env.catch_z, x_limit=env.x_limit,
                        dodge_radius=env.dodge_radius)
    prof = LatencyProfiler()
    for i in range(30):
        prof.record(run_episode(env, pol, seed=100 + i))
    s = prof.summary()
    env.close()
    assert s["success_rate"] >= 0.9
    assert s["false_trigger_rate"] == 0.0


def test_fixed_lambda_training_mode():
    """alpha_mode=fixed：α 恒为设定值，控制器记录更新但不漂移。"""
    venv = build_vec_env(seed=0, n_envs=1, alpha_mode="fixed", beta=0.01,
                         alpha_fixed=1.0)
    obs = venv.reset()
    for _ in range(400):  # 足够跑完若干 episode
        obs, _, dones, _ = venv.step(venv.action_space.sample())
    alphas = venv.env_method("get_alpha")
    logs = venv.env_method("get_alpha_log")
    venv.close()
    assert alphas == [1.0]
    assert len(logs[0]) > 0 and all(a == 1.0 for a in logs[0])


def test_dodge_dualpath_soft_handover_real():
    """避障双通路：威胁解除（球掠过）→ 仲裁器交还，软过渡真实触发。"""
    base = PaddleCatchEnv(scenario_path=DODGE_YAML, seed=0)
    env = DualPathEnv(base, CruisePlanner(v_max=base.v_max, x_limit=base.x_limit),
                      load_ppo_policy("runs/ppo_dodge_alpha"))
    prof = LatencyProfiler()
    handovers = 0
    for i in range(15):
        ev = run_dualpath_episode(env, seed=400 + i)
        prof.record(ev)
        if ev.get("arbiter", {}).get("disengagements", 0) > 0:
            handovers += 1
    s = prof.summary()
    env.close()
    assert handovers > 0, "避障场景应有真实交还（软过渡实证）"
    assert s["false_trigger_rate"] < 0.1
    assert s["success_rate"] >= 0.9


def test_speed_sweep_smoke():
    """速度扫描函数：两档 × 5 次试验，结构正确不炸。"""
    result = run_sweep(levels=[2.0, 8.0], n_trials=5, seed0=999,
                       verbose=False)
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert set(row) == {"speed", "baseline", "dual_path"}
        assert 0.0 <= row["baseline"]["success_rate"] <= 1.0
        assert 0.0 <= row["dual_path"]["success_rate"] <= 1.0
