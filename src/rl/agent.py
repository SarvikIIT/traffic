from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .traffic_env import TrafficSignalEnv

class TrafficSignalAgent:

    SUPPORTED = {"ppo", "a2c", "dqn"}

    def __init__(
        self,
        algorithm: str = "ppo",
        env: Optional[TrafficSignalEnv] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 1,
    ):
        self.algorithm = algorithm.lower()
        if self.algorithm not in self.SUPPORTED:
            raise ValueError(f"Algorithm must be one of {self.SUPPORTED}")
        self.env = env or TrafficSignalEnv()
        self.device = device
        self.policy_kwargs = policy_kwargs or {"net_arch": [256, 256]}
        self.verbose = verbose
        self._model = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _make_model(self, **kwargs):
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable_baselines3 required: pip install stable-baselines3")

        algo_cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[self.algorithm]
        self._model = algo_cls(
            "MlpPolicy",
            self.env,
            device=self.device,
            policy_kwargs=self.policy_kwargs,
            verbose=self.verbose,
            **kwargs,
        )

    def train(
        self,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        save_path: Optional[str] = None,
        log_interval: int = 10,
        eval_env: Optional[TrafficSignalEnv] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        **kwargs,
    ) -> None:
        if self._model is None:
            self._make_model(learning_rate=learning_rate, gamma=gamma, **kwargs)

        callbacks = []
        if eval_env is not None:
            try:
                from stable_baselines3.common.callbacks import EvalCallback
                eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=save_path or "models/rl_agents/",
                    log_path="logs/rl_eval/",
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                )
                callbacks.append(eval_cb)
            except Exception:
                pass

        self._model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callbacks if callbacks else None,
        )

        if save_path:
            self.save(save_path)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        if self._model is None:
            raise RuntimeError("Model not loaded or trained yet.")
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return int(action)

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        try:
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_r, std_r = evaluate_policy(
                self._model, self.env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
            )
            return {"mean_reward": float(mean_r), "std_reward": float(std_r)}
        except ImportError:
            return {"error": "stable_baselines3 not installed"}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not path.endswith(".zip"):
            path = path + ".zip"
        self._model.save(path)

    def load(self, path: str) -> None:
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable_baselines3 required")
        algo_cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[self.algorithm]
        self._model = algo_cls.load(path, env=self.env, device=self.device)

    @property
    def is_trained(self) -> bool:
        return self._model is not None
