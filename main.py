import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger


ALL_SERVICES = ["api", "video_processor", "traffic_predictor", "signal_optimizer", "dashboard"]

COMMANDS = {
    "api": [sys.executable, "-m", "uvicorn", "services.api:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload"],
    "video_processor":   [sys.executable, "services/video_processor.py"],
    "traffic_predictor": [sys.executable, "services/traffic_predictor.py"],
    "signal_optimizer":  [sys.executable, "services/signal_optimizer.py"],
    "dashboard":         [sys.executable, "-m", "streamlit", "run",
                          "services/dashboard.py", "--server.port", "8501",
                          "--server.headless", "true"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Traffic Digital Twin – Main Orchestrator")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--services", nargs="*", default=ALL_SERVICES,
                   choices=ALL_SERVICES + ["all"],
                   help="Services to start (default: all)")
    p.add_argument("--init-db", action="store_true",
                   help="Initialise database before starting")
    p.add_argument("--seed-db", action="store_true",
                   help="Seed demo data before starting")
    p.add_argument("--generate-graph", action="store_true",
                   help="Generate city graph if missing")
    return p.parse_args()


def run_subprocess(name: str, cmd: list, log) -> subprocess.Popen:
    log.info(f"Starting service: {name}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    def stream_output():
        for line in proc.stdout:
            print(f"[{name}] {line}", end="")
    t = threading.Thread(target=stream_output, daemon=True)
    t.start()
    return proc


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log  = get_logger("main")

    log.info("=" * 60)
    log.info("  City-Scale Traffic Digital Twin")
    log.info("=" * 60)

    if args.init_db or args.seed_db:
        cmd = [sys.executable, "scripts/init_database.py"]
        if args.seed_db:
            cmd.append("--seed")
        log.info("Initialising database...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Database init failed:\n{result.stderr}")
            sys.exit(1)
        log.info("Database ready.")

    if args.generate_graph:
        graph_path = "data/graphs/city_graph.json"
        if not Path(graph_path).exists():
            log.info("Generating city graph...")
            subprocess.run([sys.executable, "scripts/generate_graph.py"], check=True)
        else:
            log.info(f"Graph already exists: {graph_path}")

    services = args.services
    if "all" in services:
        services = ALL_SERVICES
    log.info(f"Starting services: {services}")

    procs = {}
    for svc in services:
        if svc not in COMMANDS:
            log.warning(f"Unknown service: {svc}")
            continue
        proc = run_subprocess(svc, COMMANDS[svc], log)
        procs[svc] = proc
        time.sleep(1)  # stagger startup

    if not procs:
        log.error("No services started.")
        sys.exit(1)

    log.info("-" * 60)
    if "api" in procs:
        log.info("API:       http://localhost:8000")
        log.info("API Docs:  http://localhost:8000/docs")
    if "dashboard" in procs:
        log.info("Dashboard: http://localhost:8501")
    log.info("-" * 60)
    log.info("Press Ctrl+C to stop all services.")

    def shutdown(sig, frame):
        log.info("\nShutting down...")
        for name, proc in procs.items():
            log.info(f"  Stopping {name}...")
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        for name, proc in procs.items():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.info("All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(10)
        for name, proc in list(procs.items()):
            if proc.poll() is not None:
                log.warning(f"Service {name} exited (code {proc.returncode}). Restarting...")
                new_proc = run_subprocess(name, COMMANDS[name], log)
                procs[name] = new_proc


if __name__ == "__main__":
    main()
