"""
Centralized logging setup for the backend.

Usage, anywhere in the codebase:

    from logger import get_logger
    log = get_logger(__name__)

    log.info("scored email", extra={"score": 0.87})
    log.warning("layer1 model not found, falling back to mock")

Configuration is read once from environment variables:
    LOG_LEVEL  - DEBUG | INFO | WARNING | ERROR (default: INFO)
    LOG_FORMAT - "text" | "json" (default: "text")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # anything passed via logging's `extra={...}` gets merged in
        reserved = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                payload[key] = value

        return json.dumps(payload, default=str)


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("LOG_FORMAT", "text").lower()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # keep noisy third-party libs quieter unless explicitly debugging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str = "sentinel_loop") -> logging.Logger:
    """Get a module-level logger, configuring the root logger on first call."""
    _configure()
    return logging.getLogger(name)