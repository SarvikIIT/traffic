import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 vehicle detector"
    )
    parser.add_argument("--data", type=str, default="data/raw",
                        help="Path to dataset YAML or video directory")
    parser.add_argument("--model", type=str, default="yolov8n",
                        help="YOLOv8 model variant (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu / cuda / 0 / auto")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use COCO pre-trained weights")
    parser.add_argument("--output", type=str, default="models/detection",
                        help="Output directory for trained weights")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()

def train(args):
    cfg = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("train_detector")

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model: {args.model}")
    if args.resume:
        model = YOLO(args.resume)
        log.info(f"Resuming from {args.resume}")
    else:
        model = YOLO(f"{args.model}.pt" if args.pretrained else f"{args.model}.yaml")

    data_path = args.data
    if not data_path.endswith(".yaml"):
        yaml_path = Path(data_path) / "dataset.yaml"
        if yaml_path.exists():
            data_path = str(yaml_path)
        else:
            log.warning(
                f"No dataset.yaml found in {data_path}. "
                "Using COCO128 as demo dataset."
            )
            data_path = "coco128.yaml"

    log.info(f"Starting training | data={data_path} epochs={args.epochs}")
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=args.device if args.device != "auto" else None,
        project=args.output,
        name="vehicle_detector",
        exist_ok=True,
        save=True,
        plots=True,
    )

    log.info(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")

    log.info("Running validation...")
    metrics = model.val()
    log.info(f"mAP@0.5: {metrics.box.map50:.4f}")
    log.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    export_path = Path(args.output) / "vehicle_detector" / "weights" / "best.pt"
    if export_path.exists():
        log.info(f"Model saved to: {export_path}")

if __name__ == "__main__":
    args = parse_args()
    train(args)
