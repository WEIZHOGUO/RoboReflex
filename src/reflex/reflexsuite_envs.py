"""ReflexSuite-mini 场景环境（复刻上交大 ReflexBench 任务精神，非官方代码）：

- WhackMoleEnv   打地鼠：地鼠随机位置探出，存活时间极短，限时击中
- RollingBallEnv 滚球拦截：球贴地滚向挡板线，挡板横向卡位拦截

与 PaddleCatchEnv 同一套约定（ARCHITECTURE.md）：
- 物理 500Hz，控制频率由 control_hz 决定（frame_skip 降采样）
- 事件双轨时间戳（stim / first_obs / first_action / outcome）
- 无目标期挡板位移 > false_trigger_threshold 记误触发
- 观测保持 8 维布局 [目标位置(3), 目标状态(3), 挡板位置(1), 挡板速度(1)]，
  与接球环境同构，PPO 策略/归一化管线零改动复用

仲裁/奖励挂钩（与 PaddleCatchEnv 相同接口）：
- ball_launched      目标是否已激活（探出/放球）
- threat_ttc(obs)    威胁时间余量：地鼠=剩余存活时间；滚球=到达挡板线时间
- get_time_window()  本试验物理时间窗（α 奖励基准）
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from .env import EV_FIRST_ACTION, EV_FIRST_OBS, EV_OUTCOME, EV_STIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _PaddleBase(gym.Env):
    """公共骨架：1-DOF 挡板（paddle_slide/paddle_vel）+ 双轨时间戳事件账本。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    success_outcome: str = ""

    def __init__(self, xml_path, scenario_path=None, scenario: dict | None = None,
                 control_hz: int = 50, seed: int | None = None,
                 render_mode: str | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        if scenario is None:
            with open(scenario_path, encoding="utf-8") as f:
                scenario = yaml.safe_load(f)
        self.config = scenario

        paddle_cfg = scenario.get("paddle", {})
        self.v_max = float(paddle_cfg.get("max_speed", 6.0))
        self.x_limit = float(paddle_cfg.get("x_limit", 0.95))
        self.max_episode_time = float(scenario.get("max_episode_time", 6.0))
        self.false_trigger_threshold = float(scenario.get("false_trigger_threshold", 0.02))
        self.action_threshold = float(scenario.get("action_threshold", 0.1))

        self.control_hz = int(control_hz)
        self.frame_skip = max(1, round(1.0 / control_hz / self.model.opt.timestep))
        self.render_mode = render_mode
        self._renderer = None

        m = self.model
        self._paddle_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "paddle_slide")]
        self._paddle_dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "paddle_slide")]
        self._paddle_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paddle_geom")
        self._paddle_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "paddle_vel")

        self.action_space = spaces.Box(-self.v_max, self.v_max, shape=(1,), dtype=np.float64)
        self._base_seed = seed
        self._reset_count = 0
        self._rng = np.random.default_rng(seed)
        self._events: dict = {}

    # ---------- 内部工具 ----------

    def _stamp(self) -> dict:
        return {"sim": float(self.data.time), "wall_ns": time.perf_counter_ns()}

    def _contact_between(self, gid_a: int, gid_b: int) -> bool:
        for i in range(self.data.ncon):
            pair = (self.data.contact[i].geom1, self.data.contact[i].geom2)
            if gid_a in pair and gid_b in pair:
                return True
        return False

    @property
    def ball_launched(self) -> bool:
        """目标是否已激活（探出/放球）。命名与接球环境一致（仲裁器通用）。"""
        raise NotImplementedError

    def threat_ttc(self, obs) -> float:
        raise NotImplementedError

    def get_time_window(self) -> float:
        raise NotImplementedError

    # ---------- 渲染 ----------

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=getattr(self, "_camera", "side"))
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class WhackMoleEnv(_PaddleBase):
    """打地鼠：地鼠（mole_free 刚体）探头存活 alive_s，挡板限时击中。

    终局：whacked（击中，成功）/ missed（存活期满未击中，失败）/ timeout。
    地鼠缩回后钉在地板下（不可接触）；episode 在首个地鼠出结果后结束。
    """

    DEFAULT_XML = PROJECT_ROOT / "assets" / "whack_a_mole.xml"

    def __init__(self, scenario_path=None, scenario: dict | None = None, **kw):
        super().__init__(kw.pop("xml_path", None) or self.DEFAULT_XML,
                         scenario_path, scenario, **kw)
        mole_cfg = self.config["mole"]
        self.delay_range = tuple(mole_cfg["delay_range"])
        self.alive_range = tuple(mole_cfg["alive_range"])
        self.mole_x_range = tuple(mole_cfg["x_range"])
        self.z_up = float(mole_cfg.get("z_up", 0.10))
        self.z_down = float(mole_cfg.get("z_down", -0.50))

        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "mole_free")
        self._mole_qadr = m.jnt_qposadr[jid]
        self._mole_dadr = m.jnt_dofadr[jid]
        self._mole_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mole_geom")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64)
        self.success_outcome = "whacked"

    # ---------- 内部 ----------

    def _pin_mole(self, z: float):
        q, d = self._mole_qadr, self._mole_dadr
        self.data.qpos[q:q + 3] = (self._mole_x, 0.0, z)
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[d:d + 6] = 0.0

    def _check_outcome(self):
        if self._up and self._contact_between(self._mole_gid, self._paddle_gid):
            return "whacked"
        if self._up and float(self.data.time) >= self._alive_until:
            return "missed"
        return None

    def _get_obs(self) -> np.ndarray:
        q = self._mole_qadr
        remaining = max(0.0, self._alive_until - float(self.data.time)) if self._up else 0.0
        # 缩回期屏蔽位置（ReflexBench 语义：探头前目标位置不可知）——
        # 否则 PPO 会从观测里读到 mole_x 提前蹲点（观测泄漏，奖励作弊）
        mx = float(self.data.qpos[q]) if self._up else 0.0
        mz = float(self.data.qpos[q + 2]) if self._up else self.z_down
        return np.array([
            mx, 0.0, mz,                                   # 地鼠位置 (x, 0, z)
            1.0 if self._up else 0.0,                        # 是否探头
            remaining,                                       # 剩余存活时间
            0.0,
            self.data.qpos[self._paddle_qadr],
            self.data.qvel[self._paddle_dadr],
        ])

    @property
    def ball_launched(self) -> bool:
        return self._up

    @property
    def mole_x(self) -> float:
        return self._mole_x

    def threat_ttc(self, obs) -> float:
        """威胁时间余量 = 剩余存活时间（越接近缩回越危险）。"""
        if not self._up:
            return math.inf
        return max(0.0, self._alive_until - float(self.data.time))

    def get_time_window(self) -> float:
        return self._alive_s

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self._base_seed is not None:
            seed = self._base_seed + self._reset_count
        self._reset_count += 1
        self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._mole_x = float(self._rng.uniform(*self.mole_x_range))
        self._pop_time = float(self._rng.uniform(*self.delay_range))
        self._alive_s = float(self._rng.uniform(*self.alive_range))
        self._alive_until = math.inf
        self._up = False
        self._pin_mole(self.z_down)
        mujoco.mj_forward(self.model, self.data)

        self._events = {}
        self._prestim_max_dev = 0.0
        self._first_action_recorded = False
        self._first_obs_recorded = False
        return self._get_obs(), {"mole_spec": {
            "mole_x": self._mole_x, "pop_time": self._pop_time,
            "alive_s": self._alive_s}}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(1),
                         -self.v_max, self.v_max)
        self.data.ctrl[self._paddle_act] = action[0]

        if self._up and not self._first_action_recorded \
                and abs(action[0]) >= self.action_threshold:
            self._events[EV_FIRST_ACTION] = self._stamp()
            self._first_action_recorded = True

        outcome = None
        for _ in range(self.frame_skip):
            if not self._up and float(self.data.time) >= self._pop_time:
                self._up = True
                self._alive_until = float(self.data.time) + self._alive_s
                self._events[EV_STIM] = self._stamp()
            self._pin_mole(self.z_up if self._up else self.z_down)
            mujoco.mj_step(self.model, self.data)
            if not self._up:
                dev = abs(float(self.data.qpos[self._paddle_qadr]))
                self._prestim_max_dev = max(self._prestim_max_dev, dev)
            outcome = self._check_outcome()
            if outcome:
                break

        obs = self._get_obs()
        if self._up and not self._first_obs_recorded:
            self._events[EV_FIRST_OBS] = self._stamp()
            self._first_obs_recorded = True

        timeout = float(self.data.time) >= self.max_episode_time
        if outcome is None and timeout:
            outcome = "timeout"
        terminated = outcome is not None

        reward = {"whacked": 1.0, "missed": -1.0, "timeout": -1.0}.get(outcome, 0.0)
        info = {"launched": self._up, "outcome": outcome}
        if terminated:
            self._events[EV_OUTCOME] = {**self._stamp(), "type": outcome}
            self._events["false_trigger"] = bool(
                self._prestim_max_dev > self.false_trigger_threshold)
            self._events["prestim_max_dev"] = float(self._prestim_max_dev)
            self._events["launch_spec"] = {
                "mole_x": self._mole_x, "pop_time": self._pop_time,
                "alive_s": self._alive_s}
            info["events"] = self._events
        return obs, reward, terminated, False, info


class RollingBallEnv(_PaddleBase):
    """滚球拦截：球贴地滚向挡板线（y=0），挡板横向卡位。

    终局：blocked（球-挡板接触，成功）/ missed（球越过 pass_y，失败）/ timeout。
    与接球环境判定的关键差异：球全程接触地板（滚动），终局只看球-挡板接触
    和越线，不看球-地板接触。
    """

    DEFAULT_XML = PROJECT_ROOT / "assets" / "rolling_ball.xml"

    def __init__(self, scenario_path=None, scenario: dict | None = None, **kw):
        super().__init__(kw.pop("xml_path", None) or self.DEFAULT_XML,
                         scenario_path, scenario, **kw)
        roll_cfg = self.config["roll"]
        self.delay_range = tuple(roll_cfg["delay_range"])
        self.speed_range = tuple(roll_cfg["speed_range"])
        self.vx_range = tuple(roll_cfg["vx_range"])
        self.spawn_x_range = tuple(roll_cfg["spawn_x_range"])
        self.spawn_y = float(roll_cfg.get("spawn_y", 2.0))
        self.pass_y = float(self.config.get("pass_y", -0.50))
        self._ball_radius = 0.05  # 与 rolling_ball.xml 的 ball_geom size 一致

        m = self.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self._ball_qadr = m.jnt_qposadr[jid]
        self._ball_dadr = m.jnt_dofadr[jid]
        self._ball_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64)
        self.success_outcome = "blocked"
        self._camera = "corner"  # 3/4 俯视：侧视看不到纵深来球

    # ---------- 内部 ----------

    def _pin_ball(self):
        q, d = self._ball_qadr, self._ball_dadr
        self.data.qpos[q:q + 3] = (self._spawn_x, self.spawn_y, self._ball_radius)
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[d:d + 6] = 0.0

    def _launch(self):
        q, d = self._ball_qadr, self._ball_dadr
        self._pin_ball()
        self.data.qvel[d:d + 3] = (self._vx, -self._speed, 0.0)
        self._launched = True
        self._events[EV_STIM] = self._stamp()

    def _check_outcome(self):
        if not self._launched:
            return None
        if self._contact_between(self._ball_gid, self._paddle_gid):
            return "blocked"
        if float(self.data.qpos[self._ball_qadr + 1]) < self.pass_y:
            return "missed"
        return None

    def _get_obs(self) -> np.ndarray:
        q, d = self._ball_qadr, self._ball_dadr
        return np.concatenate([
            self.data.qpos[q:q + 3],          # 球位置
            self.data.qvel[d:d + 3],          # 球速度
            [self.data.qpos[self._paddle_qadr]],
            [self.data.qvel[self._paddle_dadr]],
        ])

    @property
    def ball_launched(self) -> bool:
        return self._launched

    def threat_ttc(self, obs) -> float:
        """威胁时间余量 = 球按当前速度到达挡板线（y=0）的时间。"""
        if not self._launched:
            return math.inf
        by, vy = float(obs[1]), float(obs[4])
        if vy >= -1e-6 or by <= 0.0:
            return 0.0 if by <= 0.0 else math.inf
        return by / (-vy)

    def get_time_window(self) -> float:
        return self.spawn_y / self._speed

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self._base_seed is not None:
            seed = self._base_seed + self._reset_count
        self._reset_count += 1
        self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._spawn_x = float(self._rng.uniform(*self.spawn_x_range))
        self._speed = float(self._rng.uniform(*self.speed_range))
        self._vx = float(self._rng.uniform(*self.vx_range))
        self._launch_delay = float(self._rng.uniform(*self.delay_range))
        self._pin_ball()
        mujoco.mj_forward(self.model, self.data)

        self._launched = False
        self._events = {}
        self._prestim_max_dev = 0.0
        self._first_action_recorded = False
        self._first_obs_recorded = False
        return self._get_obs(), {"launch_spec": {
            "spawn_x": self._spawn_x, "speed": self._speed, "vx": self._vx,
            "launch_delay": self._launch_delay}}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(1),
                         -self.v_max, self.v_max)
        self.data.ctrl[self._paddle_act] = action[0]

        if self._launched and not self._first_action_recorded \
                and abs(action[0]) >= self.action_threshold:
            self._events[EV_FIRST_ACTION] = self._stamp()
            self._first_action_recorded = True

        outcome = None
        for _ in range(self.frame_skip):
            if not self._launched:
                if self.data.time >= self._launch_delay:
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

        reward = {"blocked": 1.0, "missed": -1.0, "timeout": -1.0}.get(outcome, 0.0)
        info = {"launched": self._launched, "outcome": outcome}
        if terminated:
            self._events[EV_OUTCOME] = {**self._stamp(), "type": outcome}
            self._events["false_trigger"] = bool(
                self._prestim_max_dev > self.false_trigger_threshold)
            self._events["prestim_max_dev"] = float(self._prestim_max_dev)
            self._events["launch_spec"] = {
                "spawn_x": self._spawn_x, "speed": self._speed,
                "vx": self._vx, "launch_delay": self._launch_delay}
            info["events"] = self._events
        return obs, reward, terminated, False, info


# 场景注册表：scenario YAML 的 env 键 → 环境类（train.py / bench 按此选类）
ENV_CLASSES = {
    "whack_mole": WhackMoleEnv,
    "rolling_ball": RollingBallEnv,
}


def env_class_for_scenario(scenario: dict):
    """按场景配置的 env 键选环境类；缺省/未知回落到接球环境。"""
    from .env import PaddleCatchEnv
    return ENV_CLASSES.get(str(scenario.get("env", "paddle_catch")), PaddleCatchEnv)
