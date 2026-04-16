import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.traffic_env import TrafficSignalEnv
from src.rl.agent import TrafficSignalAgent
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL traffic signal agent")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["ppo", "a2c", "dqn"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output", type=str, default="models/rl_agents",
                        help="Directory to save trained agent")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Run evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward", type=str, default="composite",
                        choices=["wait_time_reduction", "throughput", "composite"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    return parser.parse_args()

def baseline_evaluation(env: TrafficSignalEnv, n_episodes: int = 5) -> dict:
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            action = 1 if step % 30 == 0 else 0
            obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            done = terminated or truncated
            step += 1
        rewards.append(total_reward)
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "std_reward": (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5,
    }

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log  = get_logger("train_rl")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    env = TrafficSignalEnv(
        arrival_rates=[0.35, 0.30, 0.25, 0.20],
        reward_mode=args.reward,
        seed=args.seed,
    )
    eval_env = TrafficSignalEnv(
        arrival_rates=[0.35, 0.30, 0.25, 0.20],
        reward_mode=args.reward,
        seed=args.seed + 1,
    )

    log.info("Evaluating fixed-time baseline...")
    baseline = baseline_evaluation(eval_env)
    log.info(
        f"Fixed-time baseline → "
        f"mean={baseline['mean_reward']:.2f} ± {baseline['std_reward']:.2f}"
    )

    log.info(f"Training {args.algorithm.upper()} agent for {args.timesteps:,} steps...")
    policy_kwargs = {"net_arch": [256, 256]}
    agent = TrafficSignalAgent(
        algorithm=args.algorithm,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    tensorboard_log = "runs/rl_training" if args.tensorboard else None
    save_path = str(Path(args.output) / f"{args.algorithm}_traffic")

    agent.train(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        gamma=args.gamma,
        save_path=save_path,
        eval_env=eval_env,
        eval_freq=max(1000, args.timesteps // 20),
        n_eval_episodes=5,
    )

    log.info(f"Agent saved to {save_path}.zip")

    if args.eval:
        log.info("Evaluating trained agent...")
        results = agent.evaluate(n_episodes=args.eval_episodes)
        log.info(
            f"RL Agent → "
            f"mean={results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        improvement = (
            (results["mean_reward"] - baseline["mean_reward"])
            / abs(baseline["mean_reward"] + 1e-6) * 100
        )
        log.info(f"Improvement over fixed-time: {improvement:.1f}%")

    log.info("RL training complete.")

if __name__ == "__main__":
    main()
