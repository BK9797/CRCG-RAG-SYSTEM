"""Centralized logging configuration.

Provides a single get_logger() factory so every module logs consistently
(format, level, handler) without repeating boilerplate.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root_logger(level: str = "INFO") -> None:
    """Configure the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured logger for the given module name.

    Args:
        name: Usually __name__ of the calling module.
        level: Logging level, e.g. "INFO", "DEBUG".

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root_logger(level)
    return logging.getLogger(name)
