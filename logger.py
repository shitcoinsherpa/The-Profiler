"""
Structured logging configuration for the behavioral profiling system.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # Add color to level name
        original_levelname = record.levelname
        record.levelname = f"{color}{record.levelname:8}{reset}"

        result = super().format(record)

        # Restore original level name
        record.levelname = original_levelname

        return result


def setup_logging(
    level: str = "INFO",
    log_file: bool = True,
    console: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to log to file
        console: Whether to log to console

    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            '%(asctime)s │ %(levelname)s │ %(name)-20s │ %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_filename = LOGS_DIR / f"profiler_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class AnalysisLogger:
    """
    Specialized logger for tracking analysis pipeline events.
    Provides structured logging with case ID tracking.
    """

    def __init__(self, case_id: str = None):
        """
        Initialize analysis logger.

        Args:
            case_id: Unique case identifier for this analysis
        """
        self.logger = logging.getLogger("analysis")
        self.case_id = case_id or "NO_CASE"

    def set_case_id(self, case_id: str):
        """Set the case ID for subsequent log entries."""
        self.case_id = case_id

    def _format_message(self, message: str) -> str:
        """Add case ID prefix to message."""
        return f"[{self.case_id}] {message}"

    def stage_start(self, stage_name: str, stage_number: int):
        """Log the start of an analysis stage."""
        self.logger.info(
            self._format_message(f"STAGE {stage_number} START: {stage_name}")
        )

    def stage_complete(self, stage_name: str, stage_number: int, duration_ms: float = None):
        """Log the completion of an analysis stage."""
        msg = f"STAGE {stage_number} COMPLETE: {stage_name}"
        if duration_ms:
            msg += f" ({duration_ms:.0f}ms)"
        self.logger.info(self._format_message(msg))

    def stage_error(self, stage_name: str, stage_number: int, error: str):
        """Log an error during an analysis stage."""
        self.logger.error(
            self._format_message(f"STAGE {stage_number} ERROR: {stage_name} - {error}")
        )

    def api_call(self, model: str, endpoint: str, tokens: int = None):
        """Log an API call."""
        msg = f"API CALL: {model} -> {endpoint}"
        if tokens:
            msg += f" ({tokens} tokens)"
        self.logger.debug(self._format_message(msg))

    def api_response(self, model: str, duration_ms: float, tokens: int = None):
        """Log an API response."""
        msg = f"API RESPONSE: {model} ({duration_ms:.0f}ms)"
        if tokens:
            msg += f" [{tokens} tokens]"
        self.logger.debug(self._format_message(msg))

    def api_error(self, model: str, error: str):
        """Log an API error."""
        self.logger.error(
            self._format_message(f"API ERROR: {model} - {error}")
        )

    def video_info(self, path: str, duration: float, resolution: tuple):
        """Log video information."""
        self.logger.info(
            self._format_message(
                f"VIDEO: {path} | {duration:.1f}s | {resolution[0]}x{resolution[1]}"
            )
        )

    def analysis_start(self, video_path: str):
        """Log the start of a complete analysis."""
        self.logger.info(
            self._format_message(f"ANALYSIS START: {video_path}")
        )

    def analysis_complete(self, total_duration: float):
        """Log the completion of a complete analysis."""
        self.logger.info(
            self._format_message(f"ANALYSIS COMPLETE: Total time {total_duration:.2f}s")
        )

    def analysis_failed(self, error: str):
        """Log a failed analysis."""
        self.logger.error(
            self._format_message(f"ANALYSIS FAILED: {error}")
        )


# Initialize logging on module import
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
