"""M2 测试：α 控制器更新规则、延迟奖励 wrapper、PPO 建模+短训冒烟。"""
import numpy as np
import pytest

from src.bench.baseline import BallisticBaseline
from src.bench.policies import PPOPolicy
from src.reflex.env import PaddleCatchEnv
from src.reflex.train import build_vec_env
from src.reflex.wrappers import (AlphaController, AlphaLatencyWrapper,
                                 flight_time_to_height)


def test_flight_time_to_height():
    """1.5m 静止下落到 0.17m 约 0.52s；无解返回 nan。"""
    t = flight_time_to_height(spawn_z=1.5, vz=0.0, target_z=0.17)
    assert t == pytest.approx(np.sqrt(2 * (1.5 - 0.17) / 9.81), rel=1e-6)
    assert np.isnan(flight_time_to_height(spawn_z=0.1, vz=0.0, target_z=0.5))


def test_alpha_controller_update_rule():
    """REWARD.md 闭环规则：SR 低→α 压到 α_min；SR 高→α 爬升并夹在 α_max。"""
    c = AlphaController(sr_target=0.85, eta=0.02, window=100)
    assert c.alpha == 0.0
    for _ in range(50):
        c.update(False)
    assert c.alpha == 0.0, "成功率低于目标时 α 应压在 α_min"
    for _ in range(200):
        c.update(True)
    assert c.alpha > 0.0, "成功率高于目标后 α 应爬升"
    for _ in range(5000):
        c.update(True)
    assert c.alpha <= 2.0, "α 不得超过 α_max"
    # 冻结模式（λ=0 消融组）不更新
    c0 = AlphaController(fixed=0.0)
    for _ in range(100):
        c0.update(True)
    assert c0.alpha == 0.0 and len(c0.alpha_log) == 100


def test_wrapper_latency_reward():
    """基线策略接球时：终局奖励 = 1 − α·ratio − β·rate_penalty，α 记录进 info。"""
    env = PaddleCatchEnv(seed=0)
    envw = AlphaLatencyWrapper(env, controller=AlphaController(fixed=1.0), beta=0.01)
    policy = BallisticBaseline(v_max=env.v_max, catch_z=env.catch_z, x_limit=env.x_limit)

    caught = 0
    for i in range(5):
        obs, _ = envw.reset(seed=100 + i)
        while True:
            obs, reward, terminated, truncated, info = envw.step(
                policy.act(obs, envw.ball_launched))
            if terminated or truncated:
                break
        assert "alpha" in info and "success" in info
        if info["success"]:
            caught += 1
            assert reward < 1.0, "接住时 α·延迟惩罚应使奖励小于 1"
            assert reward > 0.5, "基线反应快，惩罚不应压垮奖励"
            assert 0.0 <= info["latency_ratio"] <= 2.0
    assert caught >= 3, f"基线在 wrapper 下应仍能接球（{caught}/5）"
    assert len(envw.controller.history) == 5
    envw.close()


def test_ppo_smoke_train(tmp_path):
    """PPO 能建模型 + 短训不炸 + 保存/加载后 PPOPolicy 可推理。"""
    from stable_baselines3 import PPO

    venv = build_vec_env(seed=0, n_envs=2, alpha_mode="dynamic", beta=0.01)
    model = PPO("MlpPolicy", venv, n_steps=128, batch_size=128, n_epochs=1,
                seed=0, policy_kwargs=dict(net_arch=[32, 32]), verbose=0)
    model.learn(total_timesteps=512)
    model.save(tmp_path / "m.zip")
    venv.save(tmp_path / "vn.pkl")
    # 训练中 wrapper 的 α 控制器应已被更新
    logs = venv.env_method("get_alpha_log")
    assert all(len(log) > 0 for log in logs)
    venv.close()

    policy = PPOPolicy(tmp_path / "m.zip", tmp_path / "vn.pkl")
    env = PaddleCatchEnv(seed=1)
    obs, _ = env.reset(seed=1)
    action = policy.act(obs, env.ball_launched)
    assert action.shape == (1,) and np.all(np.isfinite(action))
    assert abs(action[0]) <= env.v_max + 1e-6
    env.close()
