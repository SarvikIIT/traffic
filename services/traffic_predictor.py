import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import TrafficGraphBuilder
from src.graph.prediction_model import TrafficPredictor
from src.utils.config import load_config
from src.utils.db import DatabaseManager, TrafficReading, TrafficPrediction, get_db
from src.utils.logger import setup_logging, get_logger

SEQ_LEN   = 12
PRED_STEP = 5
HORIZONS  = [15, 30, 60]

def fetch_recent_features(
    db: DatabaseManager,
    node_ids: list,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    N = len(node_ids)
    F = 8
    features = np.zeros((seq_len, N, F), dtype=np.float32)
    with db.session() as s:
        for ni, nid in enumerate(node_ids):
            rows = (
                s.query(TrafficReading)
                .filter(TrafficReading.intersection_id == nid)
                .order_by(TrafficReading.timestamp.desc())
                .limit(seq_len)
                .all()
            )
            rows = list(reversed(rows))
            for ti, row in enumerate(rows[-seq_len:]):
                features[ti, ni, 0] = row.density
                features[ti, ni, 1] = row.flow_rate
                features[ti, ni, 2] = row.queue_length
                features[ti, ni, 3] = row.avg_speed
                features[ti, ni, 4] = row.congestion_level
                features[ti, ni, 5] = row.vehicle_count / 100.0
                features[ti, ni, 6] = 1.0
                features[ti, ni, 7] = 1.0
    return features

def run_service():
    cfg = load_config()
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("traffic_predictor")
    db  = get_db(f"sqlite:///{cfg.get('database.sqlite_path', 'data/traffic.db')}")
    db.create_tables()

    graph_path = cfg.get("graph_network.graph_path", "data/graphs/city_graph.json")
    model_path = cfg.get("gnn.weights_path", "models/prediction/stgcn_best.pt")
    interval   = cfg.get("graph_network.update_interval", 60)

    if not Path(graph_path).exists():
        log.warning(f"Graph not found at {graph_path}. Using synthetic grid.")
        builder = TrafficGraphBuilder.create_grid_city(rows=3, cols=3)
        Path("data/graphs").mkdir(parents=True, exist_ok=True)
        builder.save(graph_path)
    else:
        builder = TrafficGraphBuilder.load(graph_path)

    node_ids  = builder.intersections
    num_nodes = len(node_ids)
    adj       = torch.tensor(builder.get_adjacency_matrix(), dtype=torch.float32)

    predictor = None
    if Path(model_path).exists():
        predictor = TrafficPredictor.from_checkpoint(
            model_path, num_nodes=num_nodes, seq_len=SEQ_LEN
        )
        log.info(f"Loaded GNN from {model_path}")
    else:
        log.warning(f"No trained model at {model_path}. Predictions will be skipped.")

    log.info(f"Traffic predictor service started – {num_nodes} nodes, interval={interval}s")

    while True:
        if predictor is None:
            time.sleep(interval)
            continue

        features = fetch_recent_features(db, node_ids, seq_len=SEQ_LEN)
        x = torch.tensor(features, dtype=torch.float32)

        for horizon in HORIZONS:
            pred_steps = max(1, horizon // PRED_STEP)
            try:
                pred = predictor.predict(x, adj)
                step_idx = min(pred_steps - 1, pred.shape[0] - 1)
                with db.session() as s:
                    for ni, nid in enumerate(node_ids):
                        entry = TrafficPrediction(
                            intersection_id=nid,
                            horizon_minutes=horizon,
                            predicted_density=float(pred[step_idx, ni]),
                            predicted_flow=None,
                            predicted_queue=None,
                            confidence=0.85,
                        )
                        s.add(entry)
                log.info(f"Stored {len(node_ids)} predictions @ horizon={horizon}min")
            except Exception as e:
                log.error(f"Prediction failed: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    run_service()
