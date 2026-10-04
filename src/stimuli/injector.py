"""刺激注入引擎：从场景 YAML 读取发球分布，reset 时采样发球参数。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class LaunchSpec:
    """一次试验的发球参数（采样结果，不可变）。"""
    spawn_pos: np.ndarray   # (3,) 发球位置
    velocity: np.ndarray    # (3,) 初速度
    launch_delay: float     # 复位后延迟发球时间（秒）
    speed: float            # 采样到的初速（m/s）
    angle_deg: float        # 采样到的水平角度（度）


class StimulusInjector:
    """从场景配置采样发球分布。角度定义：x-z 平面内相对竖直向下方向的偏角，正值朝 +x。"""

    def __init__(self, config: dict):
        self.config = config
        launch = config["launch"]
        self.delay_range = tuple(launch["delay_range"])
        self.speed_range = tuple(launch["speed_range"])
        self.angle_range = tuple(launch["angle_range"])
        self.spawn_height = float(launch["spawn_height"])
        self.spawn_x_range = tuple(launch.get("spawn_x_range", [0.0, 0.0]))

    @classmethod
    def from_yaml(cls, path) -> "StimulusInjector":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def sample(self, rng: np.random.Generator) -> LaunchSpec:
        speed = float(rng.uniform(*self.speed_range))
        angle_deg = float(rng.uniform(*self.angle_range))
        angle = np.deg2rad(angle_deg)
        vx = speed * np.sin(angle)
        vz = -speed * np.cos(angle)  # 竖直分量恒向下
        x0 = float(rng.uniform(*self.spawn_x_range))
        return LaunchSpec(
            spawn_pos=np.array([x0, 0.0, self.spawn_height]),
            velocity=np.array([vx, 0.0, vz]),
            launch_delay=float(rng.uniform(*self.delay_range)),
            speed=speed,
            angle_deg=angle_deg,
        )
