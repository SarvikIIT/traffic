import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.dataset import TrafficGraphDataset
from src.graph.prediction_model import STGCN
from src.graph.builder import TrafficGraphBuilder
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train STGCN traffic predictor")
    parser.add_argument("--graph", type=str, default=None,
                        help="Path to city_graph.json (optional)")
    parser.add_argument("--history", type=int, default=24,
                        help="Hours of historical data used (or --seq_len in steps)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Input sequence length (time-steps)")
    parser.add_argument("--pred_len", type=int, default=3,
                        help="Prediction horizon (time-steps)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use synthetic data for demo / testing")
    parser.add_argument("--nodes", type=int, default=25,
                        help="Number of synthetic nodes")
    parser.add_argument("--output", type=str, default="models/prediction",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser.parse_args()

def get_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mape = ((pred - target).abs() / (target.abs() + 1e-6)).mean().item() * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}

def train_epoch(model, loader, optimizer, criterion, adj, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y, adj_d = x.to(device), y.to(device), adj.to(device)
        optimizer.zero_grad()
        pred = model(x, adj_d)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, criterion, adj, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x, y, _ in loader:
        x, y, adj_d = x.to(device), y.to(device), adj.to(device)
        pred = model(x, adj_d)
        total_loss += criterion(pred, y).item()
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    preds = torch.cat(all_preds)
    tgts  = torch.cat(all_targets)
    metrics = compute_metrics(preds, tgts)
    metrics["loss"] = total_loss / len(loader)
    return metrics

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log  = get_logger("train_gnn")
    device = get_device(args.device)
    log.info(f"Device: {device}")

    if args.synthetic or args.graph is None:
        log.info(f"Generating synthetic dataset (nodes={args.nodes})")
        train_ds, val_ds, test_ds = TrafficGraphDataset.generate_synthetic(
            num_nodes=args.nodes,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
        )
        num_nodes = args.nodes
        edge_index = train_ds.edge_index
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(edge_index.shape[1]):
            adj[edge_index[0, i], edge_index[1, i]] = 1.0
    else:
        log.info(f"Loading graph from {args.graph}")
        builder = TrafficGraphBuilder.load(args.graph)
        num_nodes = builder.num_nodes
        adj_np = builder.get_adjacency_matrix()
        adj = torch.tensor(adj_np, dtype=torch.float32)
        feat_path = Path("data/processed/node_features.npy")
        if not feat_path.exists():
            log.warning("No processed features found. Using synthetic fallback.")
            train_ds, val_ds, test_ds = TrafficGraphDataset.generate_synthetic(
                num_nodes=num_nodes, seq_len=args.seq_len, pred_len=args.pred_len
            )
        else:
            features = np.load(str(feat_path))
            edge_index = builder.get_edge_index()
            T = len(features)
            t_end, v_end = int(T * 0.7), int(T * 0.8)
            mean = features[:t_end].mean(axis=(0, 1), keepdims=True)
            std  = features[:t_end].std(axis=(0, 1), keepdims=True) + 1e-8
            train_ds = TrafficGraphDataset(features[:t_end], edge_index,
                                           args.seq_len, args.pred_len,
                                           train_mean=mean, train_std=std)
            val_ds   = TrafficGraphDataset(features[t_end:v_end], edge_index,
                                           args.seq_len, args.pred_len,
                                           train_mean=mean, train_std=std)
            test_ds  = TrafficGraphDataset(features[v_end:], edge_index,
                                           args.seq_len, args.pred_len,
                                           train_mean=mean, train_std=std)

    in_channels = train_ds[0][0].shape[-1]
    log.info(f"Dataset: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    log.info(f"Nodes={num_nodes} | In-channels={in_channels} | "
             f"seq_len={args.seq_len} | pred_len={args.pred_len}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        pred_len=args.pred_len,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)
    adj = adj.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="traffic-digital-twin", name="stgcn")
            wandb.config.update(vars(args))
        except ImportError:
            log.warning("wandb not installed; skipping.")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    best_epoch = 0
    patience = 30
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, adj, device)
        val_metrics = eval_epoch(model, val_loader, criterion, adj, device)
        scheduler.step()

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if wandb_run:
            wandb_run.log({"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(ckpt, Path(args.output) / "stgcn_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    ckpt = torch.load(Path(args.output) / "stgcn_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = eval_epoch(model, test_loader, criterion, adj, device)
    log.info(
        f"Test results → MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"MAPE: {test_metrics['mape']:.2f}%"
    )

    if wandb_run:
        wandb_run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    log.info(f"Training complete. Best model: {Path(args.output) / 'stgcn_best.pt'}")

if __name__ == "__main__":
    main()
