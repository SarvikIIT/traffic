from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .reward import IntersectionState, compute_reward

class Vehicle:
    __slots__ = ["approach", "arrival_time", "wait_time", "served"]

    def __init__(self, approach: int, arrival_time: float):
        self.approach = approach
        self.arrival_time = arrival_time
        self.wait_time = 0.0
        self.served = False

class TrafficSignalEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    PHASE_GREEN = {
        0: [0, 1],
        1: [2, 3],
    }
    NUM_PHASES = 2
    NUM_APPROACHES = 4

    def __init__(
        self,
        arrival_rates: Optional[List[float]] = None,
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_time: float = 3.0,
        sim_step: float = 1.0,
        max_steps: int = 3600,
        reward_mode: str = "composite",
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.arrival_rates = arrival_rates or [0.3, 0.3, 0.2, 0.2]
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.sim_step = sim_step
        self.max_steps = max_steps
        self.reward_mode = reward_mode

        obs_dim = 4 + 4 + self.NUM_PHASES + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        self._rng = np.random.default_rng(seed)
        self._vehicles: List[Vehicle] = []
        self._reset_state()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        prev_state = self._capture_state()

        if action == 1:
            if self._time_in_phase >= self.min_green:
                self._do_yellow_transition()
                self._current_phase = 1 - self._current_phase
                self._time_in_phase = 0.0

        if self._time_in_phase >= self.max_green:
            self._do_yellow_transition()
            self._current_phase = 1 - self._current_phase
            self._time_in_phase = 0.0

        self._simulate_step()
        self._time_in_phase += self.sim_step
        self._elapsed += self.sim_step
        self._step_count += 1

        curr_state = self._capture_state()
        reward = compute_reward(curr_state, prev_state, mode=self.reward_mode)

        obs = self._get_obs()
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {
            "total_wait": self._total_wait,
            "total_served": self._total_served,
            "queue_lengths": self._queue.copy(),
            "phase": self._current_phase,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        q = self._queue
        print(
            f"Step {self._step_count:5d} | Phase {self._current_phase} "
            f"({self._time_in_phase:.0f}s) | "
            f"Queues N={q[0]:.0f} S={q[1]:.0f} E={q[2]:.0f} W={q[3]:.0f} | "
            f"Served {self._total_served} | Wait {self._total_wait:.1f}s"
        )

    def _reset_state(self) -> None:
        self._vehicles = []
        self._queue = np.zeros(self.NUM_APPROACHES, dtype=np.float32)
        self._wait = np.zeros(self.NUM_APPROACHES, dtype=np.float32)
        self._throughput = np.zeros(self.NUM_APPROACHES, dtype=np.float32)
        self._current_phase = 0
        self._time_in_phase = 0.0
        self._elapsed = 0.0
        self._step_count = 0
        self._total_wait = 0.0
        self._total_served = 0

        self._hour = self._rng.integers(0, 24)
        self._dow = self._rng.integers(0, 7)

    def _simulate_step(self) -> None:
        t = self._elapsed
        for approach in range(self.NUM_APPROACHES):
            rate = self._arrival_rate(approach)
            n_arrivals = self._rng.poisson(rate * self.sim_step)
            for _ in range(n_arrivals):
                self._vehicles.append(Vehicle(approach, t))
                self._queue[approach] += 1

        sat_flow = 0.5
        green_approaches = self.PHASE_GREEN[self._current_phase]
        for approach in green_approaches:
            n_depart = int(sat_flow * self.sim_step)
            departed = 0
            for v in self._vehicles:
                if v.approach == approach and not v.served and departed < n_depart:
                    v.served = True
                    v.wait_time = t - v.arrival_time
                    self._total_wait += v.wait_time
                    self._total_served += 1
                    self._queue[approach] = max(0, self._queue[approach] - 1)
                    self._throughput[approach] += 1
                    departed += 1

        for v in self._vehicles:
            if not v.served:
                self._wait[v.approach] += self.sim_step

        if len(self._vehicles) > 2000:
            self._vehicles = [v for v in self._vehicles if not v.served]

    def _do_yellow_transition(self) -> None:
        self._elapsed += self.yellow_time
        self._step_count += 1

    def _arrival_rate(self, approach: int) -> float:
        base = self.arrival_rates[approach]
        hour = (self._hour + self._elapsed / 3600) % 24
        am_peak = math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        pm_peak = math.exp(-0.5 * ((hour - 17) / 1.5) ** 2)
        multiplier = 1.0 + 1.5 * max(am_peak, pm_peak)
        return base * multiplier

    def _capture_state(self) -> IntersectionState:
        return IntersectionState(
            queue_lengths=self._queue.copy(),
            wait_times=self._wait.copy(),
            throughput=self._throughput.copy(),
            green_phase=self._current_phase,
            cycle_length=self._time_in_phase,
            density=self._queue.sum() / 100.0,
        )

    def _get_obs(self) -> np.ndarray:
        max_queue = 50.0
        max_wait = 300.0
        phase_onehot = np.zeros(self.NUM_PHASES, dtype=np.float32)
        phase_onehot[self._current_phase] = 1.0
        hour = (self._hour + self._elapsed / 3600) % 24
        obs = np.concatenate([
            np.clip(self._queue / max_queue, 0, 1),
            np.clip(self._wait / max_wait, 0, 1),
            phase_onehot,
            [self._time_in_phase / self.max_green],
            [hour / 24.0],
            [self._dow / 7.0],
        ]).astype(np.float32)
        return obs
