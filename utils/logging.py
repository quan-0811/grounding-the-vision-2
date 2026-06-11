"""
Logging helpers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


def setup_logger(
    name: str = "phg",
    level: int = logging.INFO,
    log_file: Optional[PathLike] = None,
) -> logging.Logger:
    """
    Create console/file logger.

    Example:
        logger = setup_logger("phg", log_file="outputs/logs/run.log")
        logger.info("hello")
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers when rerunning notebooks/scripts.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "phg") -> logging.Logger:
    return logging.getLogger(name)