#!/usr/bin/env python3
"""
Enhanced Logger with Rich Console Output, File Logging, Error Throttling, and Metrics Collection
"""

from collections import defaultdict
from datetime import datetime
from functools import wraps
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available, using basic logging")


class MetricsCollector:
    """Collects and saves pipeline metrics in JSONL format"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.metrics: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def add_metric(
        self,
        step: str,
        duration_ms: float,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        status: str = "success",
        fallback: bool = False,
        channel: Optional[str] = None,
        **kwargs,
    ):
        """Add a metric entry"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "duration_ms": round(duration_ms, 2),
            "input_path": input_path,
            "output_path": output_path,
            "status": status,
            "fallback": fallback,
            "channel": channel,
            **kwargs,
        }

        with self.lock:
            self.metrics.append(metric)

    def save_metrics(self, channel: str, date_str: Optional[str] = None) -> str:
        """Save metrics to JSONL file"""
        if not date_str:
            date_str = datetime.now().strftime("%Y%m%d")

        # Create directory structure: outputs/<kanal>/<tarih>/
        metrics_dir = self.output_dir / channel / date_str
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = metrics_dir / "metrics.jsonl"

        with self.lock:
            if self.metrics:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    for metric in self.metrics:
                        f.write(json.dumps(metric, ensure_ascii=False) + "\n")

                print(
                    f"📊 Metrics saved: {len(self.metrics)} entries to {metrics_file}"
                )
                return str(metrics_file)

        return ""

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics:
            return {}

        total_duration = sum(m.get("duration_ms", 0) for m in self.metrics)
        steps = defaultdict(list)

        for metric in self.metrics:
            step = metric.get("step", "unknown")
            steps[step].append(metric)

        summary = {
            "total_entries": len(self.metrics),
            "total_duration_ms": round(total_duration, 2),
            "steps": {step: len(metrics) for step, metrics in steps.items()},
            "status_counts": defaultdict(int),
            "fallback_count": sum(1 for m in self.metrics if m.get("fallback", False)),
        }

        for metric in self.metrics:
            status = metric.get("status", "unknown")
            summary["status_counts"][status] += 1

        return summary


class ErrorThrottler:
    """Prevents spam of repeated error messages"""

    def __init__(self, throttle_seconds: int = 60):
        self.throttle_seconds = throttle_seconds
        self.error_counts: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def should_log_error(self, error_key: str, error_message: str) -> bool:
        """Check if error should be logged (not throttled)"""
        current_time = time.time()

        with self.lock:
            if error_key not in self.error_counts:
                self.error_counts[error_key] = {
                    "count": 1,
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "message": error_message,
                }
                return True

            error_info = self.error_counts[error_key]
            time_since_last = current_time - error_info["last_seen"]

            if time_since_last >= self.throttle_seconds:
                # Reset throttling
                error_info["count"] = 1
                error_info["first_seen"] = current_time
                error_info["last_seen"] = current_time
                error_info["message"] = error_message
                return True

            # Increment count and update last seen
            error_info["count"] += 1
            error_info["last_seen"] = current_time

            # Only log if it's the first occurrence or every 10th occurrence
            return error_info["count"] == 1 or error_info["count"] % 10 == 0

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of throttled errors"""
        with self.lock:
            return {
                error_key: {
                    "count": info["count"],
                    "first_seen": datetime.fromtimestamp(
                        info["first_seen"]
                    ).isoformat(),
                    "last_seen": datetime.fromtimestamp(info["last_seen"]).isoformat(),
                    "message": info["message"],
                }
                for error_key, info in self.error_counts.items()
            }


class EnhancedLogger:
    """Enhanced logger with rich console, file logging, and metrics"""

    def __init__(
        self,
        name: str = "ProjectChimera",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize rich console
        if RICH_AVAILABLE:
            self.console = Console()
            self.rich_handler = RichHandler(
                console=self.console, show_time=True, show_path=False
            )
        else:
            self.console = None
            self.rich_handler = None

        # Initialize metrics collector
        self.metrics = MetricsCollector()

        # Initialize error throttler
        self.error_throttler = ErrorThrottler()

        # Setup logging
        self.setup_logging(console_level, file_level)

        # Progress tracking
        self.current_progress = None
        self.progress_lock = threading.Lock()

    def setup_logging(self, console_level: int, file_level: int):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Console handler
        if self.rich_handler:
            self.rich_handler.setLevel(console_level)
            self.logger.addHandler(self.rich_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        if self.console:
            self.console.print(f"📝 Logging to: {log_file}", style="blue")

    def log_metric(
        self,
        step: str,
        duration_ms: float,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        status: str = "success",
        fallback: bool = False,
        channel: Optional[str] = None,
        **kwargs,
    ):
        """Log a metric entry"""
        self.metrics.add_metric(
            step,
            duration_ms,
            input_path,
            output_path,
            status,
            fallback,
            channel,
            **kwargs,
        )

        # Log to console with rich formatting if available
        if self.console:
            status_style = (
                "green"
                if status == "success"
                else "red"
                if status == "error"
                else "yellow"
            )
            fallback_text = " (fallback)" if fallback else ""

            self.console.print(
                f"📊 {step}: {duration_ms:.1f}ms [{status}]{fallback_text}",
                style=status_style,
            )
        else:
            self.logger.info(
                f"METRIC: {step} - {duration_ms:.1f}ms - {status} - fallback:{fallback}"
            )

    def start_progress(self, description: str, total: Optional[int] = None):
        """Start a progress bar"""
        if not self.console:
            return

        with self.progress_lock:
            if self.current_progress:
                self.current_progress.stop()

            if total:
                self.current_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    console=self.console,
                )
                self.current_progress.start()
                self.current_progress.add_task(description, total=total)
            else:
                self.current_progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=self.console,
                )
                self.current_progress.start()
                self.current_progress.add_task(description, total=None)

    def update_progress(self, advance: int = 1):
        """Update progress bar"""
        if self.console and self.current_progress:
            with self.progress_lock:
                if self.current_progress.tasks:
                    self.current_progress.advance(
                        self.current_progress.tasks[0].id, advance
                    )

    def stop_progress(self):
        """Stop progress bar"""
        if self.console and self.current_progress:
            with self.progress_lock:
                self.current_progress.stop()
                self.current_progress = None

    def log_error(self, error_key: str, error_message: str, exc_info: bool = False):
        """Log error with throttling"""
        if self.error_throttler.should_log_error(error_key, error_message):
            if self.console:
                self.console.print(f"❌ {error_message}", style="red")
            self.logger.error(error_message, exc_info=exc_info)

    def log_warning(self, message: str):
        """Log warning"""
        if self.console:
            self.console.print(f"⚠️ {message}", style="yellow")
        self.logger.warning(message)

    def log_info(self, message: str):
        """Log info"""
        if self.console:
            self.console.print(f"ℹ️ {message}", style="blue")
        self.logger.info(message)

    def log_success(self, message: str):
        """Log success"""
        if self.console:
            self.console.print(f"✅ {message}", style="green")
        self.logger.info(message)

    def log_debug(self, message: str):
        """Log debug"""
        self.logger.debug(message)

    def create_metrics_table(self) -> Optional[Table]:
        """Create rich table for metrics summary"""
        if not self.console or not self.metrics.metrics:
            return None

        summary = self.metrics.get_summary()

        table = Table(title="Pipeline Metrics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Steps", str(summary.get("total_entries", 0)))
        table.add_row("Total Duration", f"{summary.get('total_duration_ms', 0):.1f}ms")
        table.add_row("Fallbacks", str(summary.get("fallback_count", 0)))

        for step, count in summary.get("steps", {}).items():
            table.add_row(f"Step: {step}", str(count))

        return table

    def display_metrics_summary(self):
        """Display metrics summary in console"""
        if not self.console:
            return

        table = self.create_metrics_table()
        if table:
            self.console.print(table)

        # Display error summary
        error_summary = self.error_throttler.get_error_summary()
        if error_summary:
            error_table = Table(title="Error Summary")
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="red")
            error_table.add_column("Last Seen", style="red")

            for error_key, info in error_summary.items():
                error_table.add_row(error_key, str(info["count"]), info["last_seen"])

            self.console.print(error_table)

    def save_metrics(self, channel: str, date_str: Optional[str] = None) -> str:
        """Save metrics and return file path"""
        return self.metrics.save_metrics(channel, date_str)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return self.metrics.get_summary()

    def clear_metrics(self):
        """Clear current metrics"""
        with self.metrics.lock:
            self.metrics.metrics.clear()


def timing_decorator(logger: EnhancedLogger, step_name: str):
    """Decorator to automatically time and log function execution"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.log_metric(step_name, duration_ms, status="success")
                return result
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                logger.log_metric(step_name, duration_ms, status="error", error=str(e))
                raise

        return wrapper

    return decorator


# Global logger instance
_global_logger = None


def get_logger(name: str = "ProjectChimera") -> EnhancedLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EnhancedLogger(name)
    return _global_logger


def setup_logger(
    name: str = "ProjectChimera",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> EnhancedLogger:
    """Setup and return logger instance"""
    global _global_logger
    _global_logger = EnhancedLogger(name, log_dir, console_level, file_level)
    return _global_logger


# Convenience functions
def log_metric(step: str, duration_ms: float, **kwargs):
    """Log a metric using global logger"""
    logger = get_logger()
    logger.log_metric(step, duration_ms, **kwargs)


def log_error(error_key: str, error_message: str, exc_info: bool = False):
    """Log error using global logger"""
    logger = get_logger()
    logger.log_error(error_key, error_message, exc_info)


def log_info(message: str):
    """Log info using global logger"""
    logger = get_logger()
    logger.log_info(message)


def log_success(message: str):
    """Log success using global logger"""
    logger = get_logger()
    logger.log_success(message)


def log_warning(message: str):
    """Log warning using global logger"""
    logger = get_logger()
    logger.log_warning(message)


if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("TestLogger", "test_logs")

    logger.log_info("Logger initialized successfully")
    logger.log_success("Test completed")
    logger.log_warning("This is a test warning")

    # Test metrics
    logger.log_metric("test_step", 150.5, status="success")
    logger.log_metric("error_step", 75.2, status="error", error="test error")

    # Test progress
    logger.start_progress("Processing items...", total=10)
    for i in range(10):
        time.sleep(0.1)
        logger.update_progress()
    logger.stop_progress()

    # Display summary
    logger.display_metrics_summary()

    # Save metrics
    logger.save_metrics("test_channel", "20241201")
