import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.prediction_model import STGCN, TrafficPredictor
from src.graph.builder import TrafficGraphBuilder
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Predict traffic congestion")
    parser.add_argument("--intersection_id", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Predict for all intersections")
    parser.add_argument("--horizon", type=int, default=30,
                        help="Prediction horizon in minutes")
    parser.add_argument("--model", type=str, default="models/prediction/stgcn_best.pt")
    parser.add_argument("--graph", type=str, default="data/graphs/city_graph.json")
    parser.add_argument("--features", type=str, default="data/processed/node_features.npy",
                        help="Historical feature array (.npy)")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data")
    parser.add_argument("--output", type=str, default=None,
                        help="Save predictions to JSON file")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()

def run_demo_prediction(seq_len: int = 12, pred_len: int = 3, num_nodes: int = 25):
    rng = np.random.default_rng(0)
    features = rng.random((seq_len, num_nodes, 8)).astype(np.float32)
    adj = np.eye(num_nodes, dtype=np.float32)
    rows = cols = 5
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                adj[n, n + 1] = adj[n + 1, n] = 1.0
            if r + 1 < rows:
                adj[n, n + rows] = adj[n + rows, n] = 1.0

    model = STGCN(num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len)
    predictor = TrafficPredictor(model)
    x = torch.tensor(features)
    a = torch.tensor(adj)
    pred = predictor.predict(x, a)
    return pred.numpy(), [f"INT_{i:03d}" for i in range(num_nodes)]

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log  = get_logger("predict_traffic")

    step_minutes = 5
    pred_len = max(1, args.horizon // step_minutes)

    if args.demo:
        log.info("Running demo prediction with synthetic data...")
        predictions, node_ids = run_demo_prediction(
            seq_len=args.seq_len, pred_len=pred_len
        )
        log.info(f"Predictions shape: {predictions.shape}")
        for i, nid in enumerate(node_ids[:5]):
            vals = predictions[:, i]
            log.info(f"  {nid}: {[round(float(v), 3) for v in vals]}")
        return

    if not Path(args.graph).exists():
        log.error(f"Graph file not found: {args.graph}. Run scripts/generate_graph.py first.")
        sys.exit(1)
    builder = TrafficGraphBuilder.load(args.graph)
    num_nodes = builder.num_nodes
    adj = torch.tensor(builder.get_adjacency_matrix(), dtype=torch.float32)
    node_ids = builder.intersections
    log.info(f"Loaded graph: {num_nodes} nodes")

    if not Path(args.model).exists():
        log.error(f"Model not found: {args.model}. Train first with train/train_gnn.py")
        sys.exit(1)
    predictor = TrafficPredictor.from_checkpoint(
        args.model,
        num_nodes=num_nodes,
        pred_len=pred_len,
        seq_len=args.seq_len,
        device=args.device,
    )
    log.info(f"Loaded model from {args.model}")
    import torch as _torch
    _ckpt_args = _torch.load(args.model, map_location="cpu", weights_only=False).get("args", {})
    pred_len = _ckpt_args.get("pred_len", pred_len)
    actual_horizon = pred_len * 5
    log.info(f"Model pred_len={pred_len} ({actual_horizon} min horizon)")

    if not Path(args.features).exists():
        log.error(f"Feature file not found: {args.features}")
        sys.exit(1)
    feat = np.load(args.features)
    seq_len = _ckpt_args.get("seq_len", args.seq_len)
    window = feat[-seq_len:]
    x = torch.tensor(window, dtype=torch.float32)

    pred = predictor.predict(x, adj)
    log.info(f"Prediction horizon: {actual_horizon} min | steps: {pred_len}")

    results = {}
    for i, nid in enumerate(node_ids):
        if args.intersection_id and nid != args.intersection_id:
            continue
        pred_vals = [round(float(pred[t, i]), 4) for t in range(pred_len)]
        results[nid] = {
            "predicted_density": pred_vals,
            "horizon_minutes": args.horizon,
            "step_minutes": step_minutes,
        }
        log.info(f"  {nid}: density={pred_vals}")

    if not results:
        log.warning(f"Intersection '{args.intersection_id}' not found in graph.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Predictions saved to {args.output}")

    return results

if __name__ == "__main__":
    main()
