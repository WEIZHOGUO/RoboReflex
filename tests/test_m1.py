"""M1 基本测试：环境不炸、profiler 结构正确、手写基线能接球。"""
import numpy as np
import pytest

from src.bench.baseline import BallisticBaseline, run_episode
from src.latency.profiler import LatencyProfiler
from src.reflex.env import PaddleCatchEnv


def test_env_reset_step_smoke():
    """环境 reset/step 全流程不炸，观测维度正确，事件结构完整。"""
    env = PaddleCatchEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert np.all(np.isfinite(obs))
    terminated = False
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        assert obs.shape == (8,) and np.all(np.isfinite(obs))
        if terminated or truncated:
            break
    assert terminated, "episode 应在超时前终止"
    events = info["events"]
    # 零动作下球必落地失败，且刺激/终局时间戳齐全（双轨）
    assert events["outcome"]["type"] in ("failed", "caught", "timeout")
    assert "sim" in events["stim"] and "wall_ns" in events["stim"]
    assert events["outcome"]["sim"] > events["stim"]["sim"]
    env.close()


def test_profiler_summary_structure(tmp_path):
    """profiler 汇总结构正确，能存取 JSON。"""
    prof = LatencyProfiler()
    for i in range(5):
        t0 = 1_000_000_000 + i * 1000
        prof.record({
            "stim": {"sim": 0.5, "wall_ns": t0},
            "first_obs": {"sim": 0.52, "wall_ns": t0 + 100_000},
            "first_action": {"sim": 0.6, "wall_ns": t0 + 200_000},
            "outcome": {"sim": 1.1, "wall_ns": t0 + 500_000,
                        "type": "caught" if i % 2 == 0 else "failed"},
            "false_trigger": i == 4,
        })
    s = prof.summary()
    assert s["n_trials"] == 5
    assert s["n_success"] == 3
    assert s["success_rate"] == pytest.approx(0.6)
    assert s["false_trigger_rate"] == pytest.approx(0.2)
    for key in ("reaction_sim_s", "reaction_wall_s", "flight_time_s"):
        assert set(s[key]) >= {"mean", "p50", "p95", "p99"}
    assert s["reaction_sim_s"]["p50"] == pytest.approx(0.1)
    # JSON 往返
    path = prof.save(tmp_path / "latency.json")
    loaded = LatencyProfiler.load(path)
    assert loaded.summary()["n_trials"] == 5
    path.unlink()


def test_baseline_catches():
    """手写弹道基线 10 次试验成功率应显著过半。"""
    env = PaddleCatchEnv(seed=42)
    policy = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z, x_limit=env.x_limit)
    prof = LatencyProfiler()
    for i in range(10):
        prof.record(run_episode(env, policy, seed=42 + i))
    s = prof.summary()
    env.close()
    assert s["success_rate"] >= 0.6, f"基线成功率过低: {s['success_rate']}"
    assert s["false_trigger_rate"] == 0.0, "基线发球前不动，不应误触发"
