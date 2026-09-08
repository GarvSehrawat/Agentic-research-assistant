"""Structured logging configuration.

We log operational metadata (requests, sessions, tool calls, durations,
statuses, errors) but never secrets, and never the model's internal
chain-of-thought.
"""
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole application."""
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported during tests).
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "httpcore", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
