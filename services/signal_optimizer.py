import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.agent import TrafficSignalAgent
from src.rl.traffic_env import TrafficSignalEnv
from src.utils.config import load_config
from src.utils.db import DatabaseManager, TrafficReading, SignalState, get_db
from src.utils.logger import setup_logging, get_logger

def build_observation(readings: list, env: TrafficSignalEnv) -> np.ndarray:
    obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    if not readings:
        return obs

    r = readings[0]
    max_queue = 50.0
    max_wait = 300.0

    q_estimate = min(r.density * 10, max_queue)
    obs[0] = q_estimate / max_queue
    obs[1] = q_estimate / max_queue
    obs[2] = q_estimate * 0.7 / max_queue
    obs[3] = q_estimate * 0.7 / max_queue

    wait_estimate = min(r.congestion_level * max_wait, max_wait)
    obs[4] = wait_estimate / max_wait
    obs[5] = wait_estimate / max_wait
    obs[6] = wait_estimate * 0.7 / max_wait
    obs[7] = wait_estimate * 0.7 / max_wait

    obs[8] = 1.0
    obs[9] = 0.0
    obs[10] = 0.5

    ts = r.timestamp
    obs[11] = ts.hour / 24.0
    obs[12] = ts.weekday() / 7.0
    return obs

def run_service():
    cfg = load_config()
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("signal_optimizer")
    db  = get_db(f"sqlite:///{cfg.get('database.sqlite_path', 'data/traffic.db')}")
    db.create_tables()

    agent_path = cfg.get("reinforcement_learning.weights_path",
                         "models/rl_agents/ppo_traffic.zip")
    interval   = cfg.get("graph_network.update_interval", 60)
    algorithm  = cfg.get("reinforcement_learning.algorithm", "ppo")
    min_green  = cfg.get("reinforcement_learning.min_green_time", 10)
    max_green  = cfg.get("reinforcement_learning.max_green_time", 60)

    env   = TrafficSignalEnv(min_green=min_green, max_green=max_green)
    agent = None
    if Path(f"{agent_path}.zip").exists() or Path(agent_path).exists():
        try:
            agent = TrafficSignalAgent(algorithm=algorithm, env=env,
                                       model_path=agent_path)
            log.info(f"Loaded RL agent from {agent_path}")
        except Exception as e:
            log.warning(f"Could not load agent: {e}. Using heuristic fallback.")
    else:
        log.warning(f"No agent at {agent_path}. Using heuristic fallback.")

    log.info(f"Signal optimizer service started, interval={interval}s")

    while True:
        with db.session() as s:
            intersections = (
                s.query(TrafficReading.intersection_id)
                .distinct()
                .all()
            )
            iids = [row[0] for row in intersections]

        for iid in iids:
            with db.session() as s:
                readings = (
                    s.query(TrafficReading)
                    .filter(TrafficReading.intersection_id == iid)
                    .order_by(TrafficReading.timestamp.desc())
                    .limit(1)
                    .all()
                )

            if not readings:
                continue

            if agent and agent.is_trained:
                obs = build_observation(readings, env)
                action = agent.predict(obs, deterministic=True)
            else:
                density = readings[0].density
                action = 1 if density > 0.05 else 0

            if action == 1:
                green_ns = max(min_green, min(max_green, int(40 + readings[0].density * 20)))
                green_ew = max(min_green, 60 - green_ns)
                phase = "EW_GREEN"
            else:
                green_ns = 35
                green_ew = 25
                phase = "NS_GREEN"

            cycle = green_ns + green_ew + 6
            with db.session() as s:
                sig = SignalState(
                    intersection_id=iid,
                    phase=phase,
                    green_duration=green_ns,
                    cycle_length=cycle,
                    optimized=1 if agent else 0,
                )
                s.add(sig)

        if iids:
            log.info(f"Optimized signals for {len(iids)} intersections.")
        time.sleep(interval)

if __name__ == "__main__":
    run_service()
