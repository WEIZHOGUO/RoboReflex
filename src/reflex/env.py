"""Gymnasium 环境：平面挡板接球（机映 RoboReflex M1）。

观测 = [球位置(3), 球速度(3), 挡板位置(1), 挡板速度(1)]，共 8 维；
动作 = 挡板目标速度（1 维连续，经 velocity 执行器跟踪）。
物理 500Hz 步进，控制频率由 control_hz 决定（frame_skip 降采样）。

事件双轨时间戳：每个关键事件同时记录 MuJoCo 逻辑时间（可复现）
与 time.perf_counter_ns 墙钟时间（真实感），供 LatencyProfiler 消费。
"""
from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from ..stimuli.injector import LaunchSpec, StimulusInjector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_ROOT / "assets" / "paddle_catch.xml"
DEFAULT_SCENARIO = PROJECT_ROOT / "src" / "scenarios" / "ball_catch_easy.yaml"

# 事件名常量
EV_STIM = "stim"              # 刺激发生（球被释放）
EV_FIRST_OBS = "first_obs"    # 刺激后首个观测就绪
EV_FIRST_ACTION = "first_action"  # 刺激后首个有效动作生效
EV_OUTCOME = "outcome"        # 终局：caught / failed / timeout


class PaddleCatchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scenario_path=None,
        scenario: dict | None = None,
        xml_path=None,
        control_hz: int = 50,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or DEFAULT_XML))
        self.data = mujoco.MjData(self.model)

        if scenario is None:
            with open(scenario_path or DEFAULT_SCENARIO, encoding="utf-8") as f:
                scenario = yaml.safe_load(f)
        self.config = scenario
        self.injector = StimulusInjector(scenario)

        paddle_cfg = scenario.get("paddle", {})
        self.v_max = float(paddle_cfg.get("max_speed", 6.0))
        self.x_limit = float(paddle_cfg.get("x_limit", 0.95))
        self.catch_z = float(scenario.get("catch_height", 0.17))
        self.max_episode_time = float(scenario.get("max_episode_time", 6.0))
        self.fail_z = float(scenario.get("fail_z", 0.03))
        self.false_trigger_threshold = float(scenario.get("false_trigger_threshold", 0.02))
        self.action_threshold = float(scenario.get("action_threshold", 0.1))
        # 场景模式：catch=接球（球-挡板接触为成功）；dodge=避障（接触为失败，
        # 球落地/掠过为成功）。威胁半径仅作场景元数据，判定仍靠接触物理。
        self.mode = str(scenario.get("mode", "catch"))
        assert self.mode in ("catch", "dodge")
        self.dodge_radius = float(scenario.get("dodge_radius", 0.22))

        self.control_hz = int(control_hz)
        self.frame_skip = max(1, round(1.0 / control_hz / self.model.opt.timestep))
        self.render_mode = render_mode
        self._renderer = None

        m = self.model
        self._paddle_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "paddle_slide")]
        self._paddle_dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "paddle_slide")]
        ball_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self._ball_qadr = m.jnt_qposadr[ball_jid]
        self._ball_dadr = m.jnt_dofadr[ball_jid]
        self._ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self._paddle_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paddle_geom")
        self._floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._paddle_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "paddle_vel")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64)
        self.action_space = spaces.Box(-self.v_max, self.v_max, shape=(1,), dtype=np.float64)

        self._base_seed = seed
        self._reset_count = 0
        self._spec: LaunchSpec | None = None
        self._rng = np.random.default_rng(seed)
        self._launched = False
        self._events: dict = {}

    # ---------- 内部工具 ----------

    def _stamp(self) -> dict:
        """双轨时间戳：逻辑时间（秒）+ 墙钟（ns）。"""
        return {"sim": float(self.data.time), "wall_ns": time.perf_counter_ns()}

    def _pin_ball(self):
        """发球前把球钉在发球点（重力下每步都会被拉走，需要步步重钉）。"""
        q, d = self._ball_qadr, self._ball_dadr
        self.data.qpos[q:q + 3] = self._spec.spawn_pos
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[d:d + 6] = 0.0

    def _launch(self):
        """到点释放：赋予采样初速并记录刺激时刻。"""
        q, d = self._ball_qadr, self._ball_dadr
        self.data.qpos[q:q + 3] = self._spec.spawn_pos
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[d:d + 6] = 0.0
        self.data.qvel[d:d + 3] = self._spec.velocity
        self._launched = True
        self._events[EV_STIM] = self._stamp()

    def _check_outcome(self):
        """contact 判定：catch 模式 球-挡板=接住/球-地板=失败；
        dodge 模式相反——球-挡板=被击中(hit)，球落地/掠过=躲开(dodged)。"""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = (con.geom1, con.geom2)
            if self._ball_gid in pair:
                if self._paddle_gid in pair:
                    return "caught" if self.mode == "catch" else "hit"
                if self._floor_gid in pair:
                    return "failed" if self.mode == "catch" else "dodged"
        if self.data.qpos[self._ball_qadr + 2] < self.fail_z:
            return "failed" if self.mode == "catch" else "dodged"
        return None

    def _get_obs(self) -> np.ndarray:
        q, d = self._ball_qadr, self._ball_dadr
        return np.concatenate([
            self.data.qpos[q:q + 3],          # 球位置
            self.data.qvel[d:d + 3],          # 球速度
            [self.data.qpos[self._paddle_qadr]],  # 挡板位置
            [self.data.qvel[self._paddle_dadr]],  # 挡板速度
        ])

    @property
    def ball_launched(self) -> bool:
        return self._launched

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self._base_seed is not None:
            seed = self._base_seed + self._reset_count
        self._reset_count += 1
        self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._spec = self.injector.sample(self._rng)
        self._pin_ball()
        mujoco.mj_forward(self.model, self.data)

        self._launched = False
        self._events = {}
        self._prestim_max_dev = 0.0
        self._first_action_recorded = False
        self._first_obs_recorded = False
        self._done_type = None
        return self._get_obs(), {"launch_spec": self._spec}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(1),
                         -self.v_max, self.v_max)
        self.data.ctrl[self._paddle_act] = action[0]

        # 首个有效动作：刺激后且超过阈值（防纹丝不动骗时间戳）
        if self._launched and not self._first_action_recorded \
                and abs(action[0]) >= self.action_threshold:
            self._events[EV_FIRST_ACTION] = self._stamp()
            self._first_action_recorded = True

        outcome = None
        for _ in range(self.frame_skip):
            if not self._launched:
                if self.data.time >= self._spec.launch_delay:
                    self._launch()
                else:
                    self._pin_ball()
            mujoco.mj_step(self.model, self.data)
            if not self._launched:
                dev = abs(float(self.data.qpos[self._paddle_qadr]))
                self._prestim_max_dev = max(self._prestim_max_dev, dev)
            outcome = self._check_outcome()
            if outcome:
                break

        obs = self._get_obs()
        if self._launched and not self._first_obs_recorded:
            self._events[EV_FIRST_OBS] = self._stamp()
            self._first_obs_recorded = True

        timeout = float(self.data.time) >= self.max_episode_time
        if outcome is None and timeout:
            outcome = "timeout"
        terminated = outcome is not None
        self._done_type = outcome

        reward = {"caught": 1.0, "failed": -1.0, "timeout": -1.0,
                  "dodged": 1.0, "hit": -1.0}.get(outcome, 0.0)
        info = {"launched": self._launched, "outcome": outcome}
        if terminated:
            self._events[EV_OUTCOME] = {**self._stamp(), "type": outcome}
            self._events["false_trigger"] = bool(
                self._prestim_max_dev > self.false_trigger_threshold)
            self._events["prestim_max_dev"] = float(self._prestim_max_dev)
            self._events["launch_spec"] = {
                "speed": self._spec.speed,
                "angle_deg": self._spec.angle_deg,
                "launch_delay": self._spec.launch_delay,
            }
            info["events"] = self._events
        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="side")
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
