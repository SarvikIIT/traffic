import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.db import (
    DatabaseManager,
    TrafficReading,
    SignalState,
    TrafficPrediction,
    Base,
)
from src.utils.logger import setup_logging, get_logger

def parse_args():
    p = argparse.ArgumentParser(description="Initialise traffic database")
    p.add_argument("--reset", action="store_true",
                   help="Drop and recreate all tables")
    p.add_argument("--seed", action="store_true",
                   help="Insert synthetic demo data after initialisation")
    p.add_argument("--nodes", type=int, default=9,
                   help="Number of synthetic intersection nodes")
    p.add_argument("--days", type=int, default=7,
                   help="Days of synthetic history to generate")
    p.add_argument("--config", type=str, default="config/config.yaml")
    return p.parse_args()

def seed_data(db: DatabaseManager, num_nodes: int = 9, days: int = 7, log=None) -> None:
    node_ids = [f"INT_{i:03d}" for i in range(num_nodes)]
    phases   = ["NS_GREEN", "EW_GREEN"]
    now      = datetime.utcnow()
    interval = timedelta(minutes=5)

    total_readings = 0
    total_signals  = 0

    with db.session() as s:
        current = now - timedelta(days=days)
        while current <= now:
            hour = current.hour
            peak_am = 8 <= hour <= 9
            peak_pm = 17 <= hour <= 18
            multiplier = 2.0 if (peak_am or peak_pm) else 1.0

            for nid in node_ids:
                density   = random.gauss(0.03 * multiplier, 0.01)
                density   = max(0.0, density)
                flow_rate = density * 20 + random.gauss(0, 0.5)
                queue_len = density * 50 + random.gauss(0, 5)
                avg_speed = max(5, 60 - density * 400 + random.gauss(0, 5))
                cong      = min(1.0, density / 0.08)
                vcount    = int(density * 100)

                s.add(TrafficReading(
                    intersection_id=nid,
                    camera_id=f"cam_{nid}",
                    timestamp=current,
                    vehicle_count=vcount,
                    density=round(density, 4),
                    avg_speed=round(avg_speed, 1),
                    queue_length=round(max(0, queue_len), 1),
                    flow_rate=round(max(0, flow_rate), 2),
                    congestion_level=round(cong, 3),
                ))
                total_readings += 1

            current += interval

        current = now - timedelta(days=days)
        while current <= now:
            for nid in node_ids:
                phase = random.choice(phases)
                g     = random.randint(20, 55)
                s.add(SignalState(
                    intersection_id=nid,
                    timestamp=current,
                    phase=phase,
                    green_duration=float(g),
                    cycle_length=float(g + random.randint(15, 40) + 6),
                    optimized=0,
                ))
                total_signals += 1
            current += timedelta(hours=1)

    if log:
        log.info(f"Seeded {total_readings} traffic readings and {total_signals} signal states.")

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log  = get_logger("init_database")

    db_type = cfg.get("database.type", "sqlite")
    if db_type == "sqlite":
        sqlite_path = cfg.get("database.sqlite_path", "data/traffic.db")
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{sqlite_path}"
    else:
        pg = cfg.section("database").get("postgresql", {})
        pw = pg.get("password") or ""
        db_url = (
            f"postgresql://{pg['user']}:{pw}"
            f"@{pg['host']}:{pg['port']}/{pg['name']}"
        )

    log.info(f"Using database: {db_url}")
    db = DatabaseManager(db_url)

    if args.reset:
        log.warning("Dropping all tables...")
        Base.metadata.drop_all(db.engine)
        log.info("Tables dropped.")

    log.info("Creating tables...")
    db.create_tables()
    log.info("Tables created successfully.")

    if args.seed:
        log.info(f"Seeding synthetic data ({args.nodes} nodes, {args.days} days)...")
        seed_data(db, num_nodes=args.nodes, days=args.days, log=log)

    log.info("Database initialisation complete.")

if __name__ == "__main__":
    main()
