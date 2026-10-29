"""Structured logging module with optional Weights & Biases integration.

Provides a centralized logging setup for the HalluGuard project with
console output, rotating file handlers, and optional W&B experiment
tracking. All production code should use this module instead of print().
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    enable_wandb: bool = False,
) -> logging.Logger:
    """Create or retrieve a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).
        log_to_file: Whether to write logs to a rotating file.
        log_file: Custom log file path. Defaults to logs/{name}.log.
        enable_wandb: Whether to attempt W&B logging integration.

    Returns:
        Configured logging.Logger instance.

    Raises:
        OSError: If log directory cannot be created.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console handler — always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optional
    if log_to_file:
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            file_path = log_file or os.path.join(LOG_DIR, f"{name.replace('.', '_')}.log")
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning("Could not create file handler: %s", e)

    # W&B integration — gracefully optional
    if enable_wandb:
        _setup_wandb_logging(logger, name)

    return logger


def _setup_wandb_logging(logger: logging.Logger, name: str) -> None:
    """Attempt to set up Weights & Biases logging integration.

    Args:
        logger: Logger instance to configure.
        name: Project name for W&B.
    """
    try:
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            logger.debug("WANDB_API_KEY not set — skipping W&B integration.")
            return

        import wandb  # noqa: F811

        if not wandb.run:
            wandb.init(
                project="halluguard",
                name=name,
                config={"module": name},
                reinit=False,
            )
        logger.info("W&B logging initialized for project 'halluguard'.")
    except ImportError:
        logger.debug("wandb not installed — skipping W&B integration.")
    except Exception as e:
        logger.warning("Failed to initialize W&B: %s", e)


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to W&B if available, otherwise log to console.

    Args:
        metrics: Dictionary of metric name -> value pairs.
        step: Optional step number for W&B logging.
    """
    logger = get_logger("metrics")
    try:
        import wandb

        if wandb.run:
            wandb.log(metrics, step=step)
            return
    except (ImportError, Exception):
        pass

    # Fallback to standard logging
    metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items())
    logger.info("Metrics: %s", metrics_str)
