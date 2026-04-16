from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass
class IntersectionState:
    queue_lengths: np.ndarray
    wait_times: np.ndarray
    throughput: np.ndarray
    green_phase: int
    cycle_length: float
    density: float

def compute_reward(
    state: IntersectionState,
    prev_state: IntersectionState,
    mode: str = "wait_time_reduction",
    weights: Dict[str, float] = None,
) -> float:
    if weights is None:
        weights = {"wait": 1.0, "queue": 0.5, "throughput": 0.3, "switch": 0.1}

    if mode == "wait_time_reduction":
        delta_wait = state.wait_times.sum() - prev_state.wait_times.sum()
        return -delta_wait

    if mode == "throughput":
        return float(state.throughput.sum())

    if mode == "composite":
        delta_wait = state.wait_times.sum() - prev_state.wait_times.sum()
        delta_queue = state.queue_lengths.sum() - prev_state.queue_lengths.sum()
        thruput = state.throughput.sum()
        phase_switch_penalty = float(state.green_phase != prev_state.green_phase)
        reward = (
            -weights.get("wait", 1.0) * delta_wait
            - weights.get("queue", 0.5) * delta_queue
            + weights.get("throughput", 0.3) * thruput
            - weights.get("switch", 0.1) * phase_switch_penalty
        )
        return float(reward)

    raise ValueError(f"Unknown reward mode: {mode}")

def pressure_reward(queue_lengths: np.ndarray, phase_pressures: np.ndarray) -> float:
    selected = phase_pressures.argmax()
    return float(phase_pressures[selected] - phase_pressures.mean())
