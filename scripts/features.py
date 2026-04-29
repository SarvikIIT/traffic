import json
import sys
import glob
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db import DatabaseManager, TrafficReading, get_db
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def load_results(results_dir: str) -> dict:
    files = sorted(glob.glob(f"{results_dir}/*_results.json"))
    data = {}
    for f in files:
        name = Path(f).stem.replace("_results", "")
        with open(f) as fp:
            data[name] = json.load(fp)
    return data

def results_to_features(results: dict, num_nodes: int = 9) -> np.ndarray:
    video_names = list(results.keys())[:num_nodes]
    max_t = 0
    series = {}
    for name in video_names:
        frames = results[name].get("frames", [])
        fps = results[name].get("fps", 30.0)
        bin_size = max(1, int(fps * 5))
        bins = []
        for i in range(0, len(frames), bin_size):
            chunk = frames[i:i + bin_size]
            avg_count   = np.mean([f["vehicle_count"] for f in chunk])
            avg_density = np.mean([f["density"] for f in chunk])
            bins.append({"count": avg_count, "density": avg_density})
        series[name] = bins
        max_t = max(max_t, len(bins))

    if max_t == 0:
        return np.zeros((100, num_nodes, 8), dtype=np.float32)

    N = len(video_names)
    features = np.zeros((max_t, N, 8), dtype=np.float32)

    for ni, name in enumerate(video_names):
        bins = series[name]
        for ti, b in enumerate(bins):
            density   = float(b["density"])
            count     = float(b["count"])
            flow      = density * 20.0
            queue     = density * 50.0
            speed     = max(5.0, 60.0 - density * 400.0)
            cong      = min(1.0, density / 0.08)
            hour_norm = 0.5
            features[ti, ni, :] = [density, flow, queue, speed, cong,
                                    count / 100.0, hour_norm, 1.0]

    for ni, name in enumerate(video_names):
        last_t = len(series[name])
        if last_t < max_t:
            features[last_t:, ni, :] = features[last_t - 1, ni, :]

    return features

def insert_to_db(results: dict, db: DatabaseManager, log) -> int:
    total = 0
    now = datetime.now(timezone.utc)
    with db.session() as s:
        for video_name, data in results.items():
            frames = data.get("frames", [])
            fps    = data.get("fps", 30.0)
            iid    = f"CAM_{video_name[:8].upper()}"
            step = max(1, int(fps))
            for i in range(0, len(frames), step):
                f = frames[i]
                density = f["density"]
                ts = now - timedelta(seconds=(len(frames) - i) / fps)
                reading = TrafficReading(
                    intersection_id=iid,
                    camera_id=video_name,
                    timestamp=ts,
                    vehicle_count=f["vehicle_count"],
                    density=round(density, 4),
                    avg_speed=max(5.0, round(60.0 - density * 400.0, 1)),
                    queue_length=round(density * 50.0, 1),
                    flow_rate=round(density * 20.0, 2),
                    congestion_level=round(min(1.0, density / 0.08), 3),
                )
                s.add(reading)
                total += 1
    return total

def main():
    cfg = load_config()
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("extract_features")

    results_dir = "results"
    results = load_results(results_dir)

    if not results:
        log.error(f"No *_results.json found in {results_dir}/")
        sys.exit(1)

    log.info(f"Found {len(results)} processed video(s): {list(results.keys())}")

    num_nodes = len(results)
    features = results_to_features(results, num_nodes=num_nodes)
    log.info(f"Feature array shape: {features.shape}  (T={features.shape[0]}, N={num_nodes}, F=8)")

    out_path = Path("data/processed/node_features.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), features)
    log.info(f"Saved node features → {out_path}")

    db = get_db(f"sqlite:///{cfg.get('database.sqlite_path', 'data/traffic.db')}")
    db.create_tables()
    n = insert_to_db(results, db, log)
    log.info(f"Inserted {n} real traffic readings into database")

    log.info("=" * 50)
    for vi, (name, data) in enumerate(results.items()):
        frames = data.get("frames", [])
        if frames:
            counts  = [f["vehicle_count"] for f in frames]
            density = [f["density"] for f in frames]
            log.info(
                f"  {name[:30]:30s} | "
                f"Frames: {len(frames):4d} | "
                f"Avg vehicles: {np.mean(counts):5.1f} | "
                f"Max: {max(counts):3d} | "
                f"Avg density: {np.mean(density):.4f}"
            )
    log.info("=" * 50)
    log.info("Feature extraction complete.")

if __name__ == "__main__":
    main()
