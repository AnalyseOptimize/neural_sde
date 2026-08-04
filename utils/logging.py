from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    log_path: Path
    fig_dir: Path
    checkpoint_dir: Path


def setup_logger(
    log_root: str | Path,
    run_name: str,
    *,
    console: bool = True,
):
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_root / f"{run_name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = log_root / f"{run_name}_{timestamp}_{suffix}"
        suffix += 1

    fig_dir = run_dir / "figs"
    checkpoint_dir = run_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "messages.txt"

    logger.remove()
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
        "{name}:{function}:{line} | {message}"
    )
    if console:
        logger.add(sys.stderr, level="INFO", format=log_format)
    logger.add(log_path, level="DEBUG", format=log_format, encoding="utf-8")

    logger.info("Logger initialized: {}", log_path)
    return logger, RunPaths(
        run_dir=run_dir,
        log_path=log_path,
        fig_dir=fig_dir,
        checkpoint_dir=checkpoint_dir,
    )


def log_config(run_logger, config: dict[str, Any]) -> None:
    run_logger.info(
        "\n========== CONFIG ==========\n{}\n============================",
        json.dumps(config, indent=2, sort_keys=True),
    )
