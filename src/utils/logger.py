import sys
from pathlib import Path
from typing import Optional

from loguru import logger

def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "traffic_twin.log",
) -> None:
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=log_level, format=fmt, colorize=True)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(log_dir) / log_file,
            level=log_level,
            format=fmt,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

def get_logger(name: str):
    return logger.bind(name=name)
