import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import TrafficGraphBuilder
from src.graph.dataset import TrafficGraphDataset
from src.graph.prediction_model import STGCN
from src.rl.traffic_env import TrafficSignalEnv
from src.rl.agent import TrafficSignalAgent
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GNN, RL, and YOLOv8 models")
    parser.add_argument("--gnn-model", type=str, default="models/prediction/stgcn_best.pt")
    parser.add_argument("--rl-model", type=str, default="models/rl_agents/ppo_traffic/best_model.zip")
    parser.add_argument("--graph", type=str, default="data/graphs/city_graph.json")
    parser.add_argument("--features", type=str, default="data/processed/node_features.npy")
    parser.add_argument("--video-dir", type=str, default="data/video")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n")
    parser.add_argument("--rl-episodes", type=int, default=10)
    parser.add_argument("--yolo-max-frames", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=3)
    parser.add_argument("--output", type=str, default="report/benchmark_results.json")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mape = ((pred - target).abs() / (target.abs() + 1e-6)).mean().item() * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def benchmark_gnn(args, device, log) -> dict:
    log.info("=" * 50)
    log.info("GNN BENCHMARK")
    log.info("=" * 50)

    gnn_path = Path(args.gnn_model)
    if not gnn_path.exists():
        log.error(f"GNN model not found: {gnn_path}")
        return {"error": f"model not found: {gnn_path}"}

    feat_path = Path(args.features)
    if not feat_path.exists():
        log.error(f"Feature file not found: {feat_path}")
        return {"error": f"features not found: {feat_path}"}

    graph_path = Path(args.graph)
    if not graph_path.exists():
        log.error(f"Graph not found: {graph_path}")
        return {"error": f"graph not found: {graph_path}"}

    builder = TrafficGraphBuilder.load(str(graph_path))
    num_nodes = builder.num_nodes
    adj_np = builder.get_adjacency_matrix()
    adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
    edge_index = builder.get_edge_index()

    features = np.load(str(feat_path))
    T = len(features)
    t_end, v_end = int(T * 0.7), int(T * 0.8)
    mean = features[:t_end].mean(axis=(0, 1), keepdims=True)
    std  = features[:t_end].std(axis=(0, 1), keepdims=True) + 1e-8
    test_ds = TrafficGraphDataset(
        features[v_end:], edge_index,
        args.seq_len, args.pred_len,
        train_mean=mean, train_std=std,
    )

    ckpt = torch.load(str(gnn_path), map_location=device, weights_only=True)
    saved_args = ckpt.get("args", {})
    hidden = saved_args.get("hidden", 64)
    layers = saved_args.get("layers", 3)
    dropout = saved_args.get("dropout", 0.1)
    in_channels = features.shape[-1]

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden,
        num_layers=layers,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    from torch.utils.data import DataLoader
    import torch.nn as nn
    criterion = nn.HuberLoss(delta=1.0)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds, all_targets = [], []
    total_loss = 0.0
    inference_times = []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y, adj_d = x.to(device), y.to(device), adj.to(device)
            t0 = time.perf_counter()
            pred = model(x, adj_d)
            inference_times.append(time.perf_counter() - t0)
            total_loss += criterion(pred, y).item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["test_loss"] = total_loss / len(loader)
    metrics["best_epoch"] = ckpt.get("epoch", "unknown")
    metrics["num_nodes"] = num_nodes
    metrics["num_params"] = total_params
    metrics["test_samples"] = len(test_ds)
    metrics["avg_inference_ms"] = float(np.mean(inference_times) * 1000)

    log.info(f"Test MAE  : {metrics['mae']:.4f}")
    log.info(f"Test RMSE : {metrics['rmse']:.4f}")
    log.info(f"Test MAPE : {metrics['mape']:.2f}%")
    log.info(f"Test Loss : {metrics['test_loss']:.4f}")
    log.info(f"Best Epoch: {metrics['best_epoch']}")
    log.info(f"Params    : {total_params:,}")
    log.info(f"Avg Batch Inference: {metrics['avg_inference_ms']:.1f} ms")
    return metrics


def _run_fixed_baseline(env: TrafficSignalEnv, n_episodes: int, log) -> dict:
    rewards = []
    steps_list = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        step = 0
        while not done:
            action = 1 if step % 30 == 0 else 0
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated
            step += 1
        rewards.append(total_r)
        steps_list.append(step)
        log.info(f"  Baseline ep {ep+1}/{n_episodes}: reward={total_r:.0f}  steps={step}")
    mean_r = float(np.mean(rewards))
    std_r  = float(np.std(rewards))
    return {
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_steps": float(np.mean(steps_list)),
        "all_rewards": rewards,
    }


def benchmark_rl(args, log) -> dict:
    log.info("=" * 50)
    log.info("RL BENCHMARK")
    log.info("=" * 50)

    rl_path = Path(args.rl_model)
    if not rl_path.exists():
        alt = Path("models/rl_agents/ppo_traffic.zip")
        if alt.exists():
            rl_path = alt
        else:
            log.error(f"RL model not found: {rl_path}")
            return {"error": f"RL model not found: {rl_path}"}

    env = TrafficSignalEnv(
        arrival_rates=[0.35, 0.30, 0.25, 0.20],
        reward_mode="composite",
        seed=999,
    )

    log.info(f"Evaluating fixed-time baseline ({args.rl_episodes} episodes)...")
    baseline = _run_fixed_baseline(env, args.rl_episodes, log)
    log.info(
        f"Baseline → mean={baseline['mean_reward']:.0f} ± {baseline['std_reward']:.0f}"
        f"  steps={baseline['mean_steps']:.0f}"
    )

    log.info(f"Loading PPO agent from {rl_path}...")
    agent = TrafficSignalAgent(
        algorithm="ppo",
        env=env,
        model_path=str(rl_path),
        device="cpu",
        verbose=0,
    )

    log.info(f"Evaluating PPO agent ({args.rl_episodes} episodes)...")
    rl_results = agent.evaluate(n_episodes=args.rl_episodes, deterministic=True)
    log.info(
        f"PPO Agent → mean={rl_results['mean_reward']:.0f} ± {rl_results['std_reward']:.0f}"
    )

    improvement = (
        (rl_results["mean_reward"] - baseline["mean_reward"])
        / (abs(baseline["mean_reward"]) + 1e-6)
    ) * 100
    log.info(f"Improvement over baseline: {improvement:.1f}%")

    return {
        "baseline_mean_reward": baseline["mean_reward"],
        "baseline_std_reward": baseline["std_reward"],
        "baseline_mean_steps": baseline["mean_steps"],
        "ppo_mean_reward": rl_results["mean_reward"],
        "ppo_std_reward": rl_results["std_reward"],
        "improvement_pct": improvement,
        "model_path": str(rl_path),
    }


def benchmark_yolo(args, log) -> dict:
    log.info("=" * 50)
    log.info("YOLO DETECTION BENCHMARK")
    log.info("=" * 50)

    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("*.avi"))
    if not videos:
        videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        log.error(f"No video files found in {video_dir}")
        return {"error": f"no videos in {video_dir}"}

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return {"error": str(e)}

    log.info(f"Loading {args.yolo_weights}...")
    model = YOLO(args.yolo_weights)

    video_path = str(videos[0])
    log.info(f"Benchmarking on: {Path(video_path).name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return {"error": f"cannot open {video_path}"}

    frame_times = []
    frame_count = 0
    detection_counts = []
    max_frames = args.yolo_max_frames

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        results = model(
            frame,
            classes=VEHICLE_CLASS_IDS,
            conf=0.40,
            iou=0.45,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        n_det = sum(len(r.boxes) for r in results)
        detection_counts.append(n_det)
        frame_count += 1

    cap.release()

    if not frame_times:
        return {"error": "no frames processed"}

    avg_ms  = float(np.mean(frame_times) * 1000)
    p50_ms  = float(np.percentile(frame_times, 50) * 1000)
    p95_ms  = float(np.percentile(frame_times, 95) * 1000)
    fps     = 1.0 / np.mean(frame_times)
    fps_p50 = 1.0 / np.percentile(frame_times, 50)
    avg_det = float(np.mean(detection_counts))

    log.info(f"Frames processed : {frame_count}")
    log.info(f"Avg latency      : {avg_ms:.1f} ms  (p50={p50_ms:.1f} ms, p95={p95_ms:.1f} ms)")
    log.info(f"Throughput       : {fps:.2f} FPS  (p50={fps_p50:.2f} FPS)")
    log.info(f"Avg detections   : {avg_det:.1f} vehicles/frame")

    return {
        "video": Path(video_path).name,
        "frames_processed": frame_count,
        "avg_latency_ms": avg_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "avg_fps": float(fps),
        "p50_fps": float(fps_p50),
        "avg_detections_per_frame": avg_det,
        "weights": args.yolo_weights,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("benchmark")
    device = get_device(args.device)
    log.info(f"Device: {device}")

    results = {}

    results["gnn"]  = benchmark_gnn(args, device, log)
    results["rl"]   = benchmark_rl(args, log)
    results["yolo"] = benchmark_yolo(args, log)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 50)
    log.info("SUMMARY")
    log.info("=" * 50)
    if "error" not in results.get("gnn", {}):
        g = results["gnn"]
        log.info(f"GNN   MAE={g['mae']:.4f}  RMSE={g['rmse']:.4f}  MAPE={g['mape']:.1f}%")
    if "error" not in results.get("rl", {}):
        r = results["rl"]
        log.info(
            f"RL    baseline={r['baseline_mean_reward']:.0f}  "
            f"ppo={r['ppo_mean_reward']:.0f}  "
            f"improvement={r['improvement_pct']:.1f}%"
        )
    if "error" not in results.get("yolo", {}):
        y = results["yolo"]
        log.info(f"YOLO  {y['avg_fps']:.1f} FPS  avg_det={y['avg_detections_per_frame']:.1f}/frame")
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
