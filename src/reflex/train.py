"""PPO 训练接球反射策略（机映 RoboReflex M2）。

奖励按 REWARD.md：R = 1[接住] − α_t·(t_response/t_window)，
α 由 AlphaController 按闭环规则逐试验更新；另加小动作变化率惩罚防抖振。
观测归一化用 VecNormalize（norm_reward=False，保持奖励量纲可解释）。

用法：
  python -m src.reflex.train --alpha-mode dynamic --timesteps 500000 --out runs/ppo_alpha
  python -m src.reflex.train --alpha-mode off     --timesteps 500000 --out runs/ppo_lambda0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import PaddleCatchEnv
from .wrappers import AlphaController, AlphaLatencyWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_vec_env(seed: int, n_envs: int, alpha_mode: str, beta: float,
                  controller_kwargs: dict | None = None,
                  scenario_path: str | None = None,
                  alpha_fixed: float = 0.5,
                  idle_beta: float = 0.0) -> VecNormalize:
    """alpha_mode: 'dynamic' -> α 闭环更新；'off' -> α=0（B 组）；
    'fixed' -> α 固定为 alpha_fixed（C 组消融档）。
    idle_beta > 0 开启空闲期动作惩罚（M4.5 奖励工程对照组）。"""
    ck = dict(controller_kwargs or {})

    def make(rank):
        def _init():
            env = PaddleCatchEnv(scenario_path=scenario_path, seed=seed + rank)
            if alpha_mode == "dynamic":
                controller = AlphaController(**ck)
            elif alpha_mode == "fixed":
                controller = AlphaController(fixed=alpha_fixed)
            else:
                controller = AlphaController(fixed=0.0)
            success_outcome = "dodged" if env.mode == "dodge" else "caught"
            return Monitor(AlphaLatencyWrapper(
                env, controller=controller, beta=beta,
                success_outcome=success_outcome, idle_beta=idle_beta))
        return _init

    venv = DummyVecEnv([make(r) for r in range(n_envs)])
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


class AlphaReportCallback(BaseCallback):
    """每 report_freq 步打印一次：当前 α、滑动窗成功率、PPO 滚动 episode 统计。"""

    def __init__(self, report_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.report_freq = int(report_freq)
        self._last_report = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_report < self.report_freq:
            return True
        self._last_report = self.num_timesteps
        alphas = self.training_env.env_method("get_alpha")
        srs = self.training_env.env_method("get_recent_sr")
        ep_rew = [ep["r"] for ep in self.model.ep_info_buffer]
        ep_len = [ep["l"] for ep in self.model.ep_info_buffer]
        print(f"[{self.num_timesteps:>9,} steps] "
              f"alpha={np.mean(alphas):.3f} recent_sr={np.mean(srs):.3f} "
              f"ep_rew_mean={np.mean(ep_rew) if ep_rew else float('nan'):+.3f} "
              f"ep_len_mean={np.mean(ep_len) if ep_len else float('nan'):.0f}",
              flush=True)
        return True


def main():
    ap = argparse.ArgumentParser(description="RoboReflex M2-M4: PPO + alpha latency reward")
    ap.add_argument("--alpha-mode", choices=["dynamic", "off", "fixed"], default="dynamic",
                    help="dynamic=α 闭环更新；off=λ=0（B 组）；fixed=固定 λ（C 组）")
    ap.add_argument("--alpha-fixed", type=float, default=0.5,
                    help="alpha_mode=fixed 时的固定 λ 值")
    ap.add_argument("--scenario", default=None,
                    help="场景 YAML（默认接球；避障传 src/scenarios/dodge_easy.yaml）")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.01, help="action rate penalty coef")
    ap.add_argument("--idle-penalty", type=float, default=0.0,
                    help="idle-period action penalty coef (M4.5 reward-engineering ablation)")
    ap.add_argument("--out", default=None, help="output dir (default runs/ppo_<mode>)")
    ap.add_argument("--report-freq", type=int, default=50_000)
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROJECT_ROOT / "runs" / f"ppo_{args.alpha_mode}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"alpha_mode={args.alpha_mode} alpha_fixed={args.alpha_fixed} "
          f"scenario={args.scenario} timesteps={args.timesteps} "
          f"seed={args.seed} out={out}", flush=True)

    venv = build_vec_env(args.seed, args.n_envs, args.alpha_mode, args.beta,
                         scenario_path=args.scenario, alpha_fixed=args.alpha_fixed,
                         idle_beta=args.idle_penalty)
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4, n_steps=1024, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=1e-3,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=args.seed, verbose=0,
    )

    callbacks = [
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=str(out / "checkpoints"),
                           name_prefix="ppo", save_vecnormalize=True),
        AlphaReportCallback(report_freq=args.report_freq),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(out / "final_model.zip")
    venv.save(out / "vecnormalize.pkl")

    # alpha 演化历史（每个并行环境各一条，dynamic 模式下才有意义）
    try:
        alpha_logs = venv.env_method("get_alpha_log")
    except Exception:
        alpha_logs = None
    venv.close()

    meta = {
        "args": vars(args),
        "alpha_log_per_env": alpha_logs,
    }
    (out / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out / 'final_model.zip'}, {out / 'vecnormalize.pkl'}", flush=True)


if __name__ == "__main__":
    main()
