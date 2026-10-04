"""PPO 策略适配器：与 BallisticBaseline 相同的 act(obs, launched) 接口。

加载 SB3 PPO 模型 + VecNormalize 观测归一化统计（pickle），
eval 时冻结归一化（training=False），确定性动作。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class PPOPolicy:
    """act(obs, launched) -> 挡板目标速度（np.ndarray shape (1,)）。"""

    def __init__(self, model_path, vecnormalize_path=None, deterministic: bool = True):
        from stable_baselines3 import PPO
        self.model = PPO.load(str(model_path))
        self.deterministic = bool(deterministic)
        self._obs_mean = None
        self._obs_std = None
        self._clip_obs = 10.0
        if vecnormalize_path is not None:
            with open(vecnormalize_path, "rb") as f:
                vn = pickle.load(f)
            self._obs_mean = np.asarray(vn.obs_rms.mean)
            self._obs_std = np.sqrt(np.asarray(vn.obs_rms.var) + vn.epsilon)
            self._clip_obs = float(vn.clip_obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self._obs_mean is None:
            return obs
        return np.clip((obs - self._obs_mean) / self._obs_std,
                       -self._clip_obs, self._clip_obs)

    def act(self, obs, launched: bool) -> np.ndarray:
        action, _ = self.model.predict(self.normalize(obs),
                                       deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float64).reshape(1)


def load_ppo_policy(run_dir) -> PPOPolicy:
    """从训练输出目录加载 final_model.zip + vecnormalize.pkl。"""
    run_dir = Path(run_dir)
    return PPOPolicy(run_dir / "final_model.zip", run_dir / "vecnormalize.pkl")
