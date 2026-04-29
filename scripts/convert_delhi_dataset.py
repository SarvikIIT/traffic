import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.db import DatabaseManager, TrafficReading, get_db
from src.utils.logger import setup_logging, get_logger

NUM_CAMERAS = 6
BIN_SECONDS = 300
CAMERA_IDS = [f"DELHI_CAM_{i}" for i in range(1, NUM_CAMERAS + 1)]


def load_all_csvs(data_dir: str, log) -> pd.DataFrame:
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not files:
        log.error(f"No CSV files found in {data_dir}")
        sys.exit(1)
    log.info(f"Loading {len(files)} CSV files...")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        log.info(f"  {Path(f).name}: {len(df):,} rows")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("EpochTime").drop_duplicates("EpochTime").reset_index(drop=True)
    log.info(f"Combined: {len(combined):,} rows | "
             f"Span: {(combined['EpochTime'].max() - combined['EpochTime'].min()) / 86400:.1f} days")
    return combined


def build_features(df: pd.DataFrame, log) -> tuple:
    df["bin"] = (df["EpochTime"] // BIN_SECONDS) * BIN_SECONDS
    bins = sorted(df["bin"].unique())
    num_timesteps = len(bins)
    log.info(f"5-min bins: {num_timesteps} | Cameras: {NUM_CAMERAS}")

    features = np.zeros((num_timesteps, NUM_CAMERAS, 8), dtype=np.float32)
    grouped = df.groupby("bin").mean(numeric_only=True)

    for ti, bin_ts in enumerate(bins):
        if bin_ts not in grouped.index:
            continue
        row = grouped.loc[bin_ts]
        hour_norm = (datetime.fromtimestamp(int(bin_ts), tz=timezone.utc).hour) / 24.0

        for cam_idx in range(NUM_CAMERAS):
            n = cam_idx + 1
            q = float(row.get(f"QueueDensity{n}", 0.0))
            s = float(row.get(f"StopDensity{n}", 0.0))

            flow       = q * 20.0
            queue_len  = s * 50.0
            speed      = max(5.0, 60.0 * (1.0 - q))
            congestion = min(1.0, q / 0.08) if q > 0 else 0.0
            count_norm = min(1.0, q * 10.0)

            features[ti, cam_idx] = [q, flow, queue_len, speed, congestion,
                                     count_norm, hour_norm, 1.0]

    for ni in range(NUM_CAMERAS):
        for fi in range(8):
            col = features[:, ni, fi]
            nz = np.nonzero(col)[0]
            if len(nz) == 0:
                continue
            for ti in range(num_timesteps):
                if col[ti] == 0 and fi not in (6, 7):
                    prev = nz[nz < ti]
                    col[ti] = col[prev[-1]] if len(prev) > 0 else col[nz[0]]
        features[:, ni, 7] = 1.0

    log.info(f"Feature array shape: {features.shape}")
    return features, bins


def write_to_db(df: pd.DataFrame, db: DatabaseManager, log) -> None:
    df["bin"] = (df["EpochTime"] // BIN_SECONDS) * BIN_SECONDS
    grouped = df.groupby("bin").mean(numeric_only=True)
    bins = sorted(grouped.index)
    log.info(f"Writing {len(bins) * NUM_CAMERAS:,} readings to database...")

    batch_size = 500
    total = 0
    with db.session() as s:
        for bi, bin_ts in enumerate(bins):
            row = grouped.loc[bin_ts]
            ts = datetime.fromtimestamp(int(bin_ts), tz=timezone.utc)

            for cam_idx in range(NUM_CAMERAS):
                n = cam_idx + 1
                q = float(row.get(f"QueueDensity{n}", 0.0))
                s_density = float(row.get(f"StopDensity{n}", 0.0))

                s.add(TrafficReading(
                    intersection_id=CAMERA_IDS[cam_idx],
                    camera_id=f"cam_{n}",
                    timestamp=ts,
                    density=round(q, 4),
                    flow_rate=round(q * 20.0, 3),
                    queue_length=round(s_density * 50.0, 2),
                    avg_speed=round(max(5.0, 60.0 * (1.0 - q)), 1),
                    congestion_level=round(min(1.0, q / 0.08) if q > 0 else 0.0, 3),
                    vehicle_count=int(q * 100),
                ))
                total += 1

            if (bi + 1) % batch_size == 0:
                log.info(f"  {bi + 1}/{len(bins)} bins written...")

    log.info(f"Inserted {total:,} readings into database.")


def main():
    parser = argparse.ArgumentParser(description="Convert Delhi Traffic Density CSV to node_features.npy")
    parser.add_argument("--data-dir", type=str,
                        default="data/DelhiTrafficDensityDataset",
                        help="Directory containing Delhi CSV files")
    parser.add_argument("--output", type=str,
                        default="data/processed/node_features.npy")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip writing to database")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("convert_delhi")

    df = load_all_csvs(args.data_dir, log)
    features, bins = build_features(df, log)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), features)
    log.info(f"Saved features → {out_path}  shape={features.shape}")

    if not args.no_db:
        db = get_db(f"sqlite:///{cfg.get('database.sqlite_path', 'data/traffic.db')}")
        db.create_tables()
        write_to_db(df, db, log)

    log.info("Conversion complete.")
    log.info(f"  Timesteps : {features.shape[0]:,}")
    log.info(f"  Cameras   : {features.shape[1]}")
    log.info(f"  Features  : {features.shape[2]}")
    log.info(f"  Days      : {features.shape[0] * BIN_SECONDS / 86400:.1f}")


if __name__ == "__main__":
    main()
