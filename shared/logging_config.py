"""Logging configuration for evaluation scripts."""
import logging
import sys


def setup_logging(script_name: str, verbose: bool = False) -> logging.Logger:
    """
    Configure logging for evaluation script.

    Args:
        script_name: Name of the script (used as logger name)
        verbose: Enable DEBUG level if True, otherwise INFO

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
