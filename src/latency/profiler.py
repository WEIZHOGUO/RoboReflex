"""延迟剖面仪：汇总每次试验的事件时间戳，输出延迟分位数 / 成功率 / 误触发率。

延迟链（每次试验）：
  刺激发生(stim) → 观测就绪(first_obs) → 首次有效动作(first_action) → 接到/失败(outcome)
逻辑时间与墙钟时间双轨统计（sim 可复现，wall 真实感）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _percentiles(values) -> dict | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class LatencyProfiler:
    """收集每次试验的事件字典（来自 PaddleCatchEnv 的 info['events']）。"""

    def __init__(self):
        self.trials: list[dict] = []

    def record(self, events: dict) -> dict:
        """录入一次试验，返回归一化后的试验记录。"""
        stim = events.get("stim")
        first_action = events.get("first_action")
        outcome = events.get("outcome")

        def _delta_sim(a, b):
            return float(a["sim"] - b["sim"]) if a and b else None

        def _delta_wall(a, b):
            return float(a["wall_ns"] - b["wall_ns"]) / 1e9 if a and b else None

        trial = {
            # 成功终局按场景命名：catch="caught"，dodge="dodged"；其余均为失败
            "success": outcome is not None and outcome.get("type") in ("caught", "dodged"),
            "outcome": outcome.get("type") if outcome else None,
            "false_trigger": bool(events.get("false_trigger", False)),
            "reaction_sim_s": _delta_sim(first_action, stim),
            "reaction_wall_s": _delta_wall(first_action, stim),
            "obs_ready_wall_s": _delta_wall(events.get("first_obs"), stim),
            "flight_time_s": _delta_sim(outcome, stim),
            "launch_spec": events.get("launch_spec"),
            # M3 五段式账本（传感/裁判/推理/执行/任务完成）+ 仲裁统计，无双通路时为空
            "ledger": events.get("ledger"),
            "arbiter": events.get("arbiter"),
        }
        self.trials.append(trial)
        return trial

    def summary(self) -> dict:
        n = len(self.trials)
        if n == 0:
            return {"n_trials": 0}
        successes = sum(t["success"] for t in self.trials)
        false_triggers = sum(t["false_trigger"] for t in self.trials)
        out = {
            "n_trials": n,
            "n_success": successes,
            "success_rate": successes / n,
            "false_trigger_rate": false_triggers / n,
            "reaction_sim_s": _percentiles([t["reaction_sim_s"] for t in self.trials]),
            "reaction_wall_s": _percentiles([t["reaction_wall_s"] for t in self.trials]),
            "obs_ready_wall_s": _percentiles([t["obs_ready_wall_s"] for t in self.trials]),
            "flight_time_s": _percentiles([t["flight_time_s"] for t in self.trials]),
        }
        # 五段式账本汇总（仅双通路组有）：各段每次试验平均耗时（ms，墙钟），裁判单列
        ledgers = [t["ledger"] for t in self.trials if t.get("ledger")]
        if ledgers:
            out["ledger_ms_mean"] = {
                seg: float(np.mean([l[seg] for l in ledgers]))
                for seg in ("sense_ms", "arbiter_ms", "inference_ms",
                            "exec_ms", "settle_ms")
            }
            out["ledger_steps_mean"] = float(np.mean([l["n_steps"] for l in ledgers]))
        arbiters = [t["arbiter"] for t in self.trials if t.get("arbiter")]
        if arbiters:
            out["arbiter_engagements_mean"] = float(
                np.mean([a["engagements"] for a in arbiters]))
            out["arbiter_false_alarms_total"] = int(
                sum(a["false_alarms"] for a in arbiters))
        return out

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "trials": self.trials}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "LatencyProfiler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        prof = cls()
        prof.trials = payload.get("trials", [])
        return prof
