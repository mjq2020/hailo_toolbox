"""
Memory management utilities for Hailo Toolbox.

This module provides comprehensive memory monitoring, optimization, and management
tools to prevent memory leaks and optimize memory usage in video inference pipelines.
"""

import gc
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from contextlib import contextmanager
from functools import wraps
import weakref
import logging

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Real-time memory monitoring utility for tracking memory usage patterns.

    This class provides comprehensive memory monitoring capabilities including
    real-time tracking, threshold warnings, and memory leak detection.
    """

    def __init__(
        self,
        warning_threshold_mb: float = 1000.0,
        critical_threshold_mb: float = 2000.0,
        monitoring_interval: float = 1.0,
        callback_warning: Optional[Callable] = None,
        callback_critical: Optional[Callable] = None,
    ):
        """
        Initialize memory monitor.

        Args:
            warning_threshold_mb: Memory threshold for warnings in MB
            critical_threshold_mb: Memory threshold for critical alerts in MB
            monitoring_interval: Monitoring interval in seconds
            callback_warning: Callback function for warning threshold
            callback_critical: Callback function for critical threshold
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        self.callback_warning = callback_warning
        self.callback_critical = callback_critical

        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None

        # Statistics
        self.start_time = None
        self.peak_memory = 0.0
        self.memory_history = []
        self.warning_count = 0
        self.critical_count = 0

    def start(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(
            f"Memory monitoring started (warning: {self.warning_threshold}MB, critical: {self.critical_threshold}MB)"
        )

    def stop(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                current_memory = self.get_current_memory_mb()
                self.peak_memory = max(self.peak_memory, current_memory)

                # Store history (keep last 1000 entries)
                timestamp = time.time() - self.start_time
                self.memory_history.append((timestamp, current_memory))
                if len(self.memory_history) > 1000:
                    self.memory_history.pop(0)

                # Check thresholds
                if current_memory > self.critical_threshold:
                    self.critical_count += 1
                    if self.callback_critical:
                        self.callback_critical(current_memory)
                    logger.critical(f"Critical memory usage: {current_memory:.1f}MB")
                elif current_memory > self.warning_threshold:
                    self.warning_count += 1
                    if self.callback_warning:
                        self.callback_warning(current_memory)
                    logger.warning(f"High memory usage: {current_memory:.1f}MB")

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            time.sleep(self.monitoring_interval)

    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory = self.get_current_memory_mb()

        stats = {
            "current_memory_mb": current_memory,
            "peak_memory_mb": self.peak_memory,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "monitoring_duration": (
                time.time() - self.start_time if self.start_time else 0
            ),
        }

        if self.memory_history:
            memories = [mem for _, mem in self.memory_history]
            stats.update(
                {
                    "avg_memory_mb": sum(memories) / len(memories),
                    "min_memory_mb": min(memories),
                    "timestamps": [t for t, _ in self.memory_history],
                    "memory_values": memories,
                }
            )

        return stats


@contextmanager
def memory_profiler(operation_name: str = "operation"):
    """
    Context manager for profiling memory usage of code blocks.

    Args:
        operation_name: Name of the operation being profiled

    Usage:
        with memory_profiler("inference"):
            # Your code here
            pass
    """
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    try:
        yield
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024
        end_time = time.time()

        memory_diff = end_memory - start_memory
        duration = end_time - start_time

        logger.info(
            f"Memory profile [{operation_name}]: "
            f"{memory_diff:+.1f}MB in {duration:.3f}s "
            f"(start: {start_memory:.1f}MB, end: {end_memory:.1f}MB)"
        )


def memory_efficient_decorator(func: Callable) -> Callable:
    """
    Decorator to enhance memory efficiency of functions.

    This decorator adds garbage collection hints and memory monitoring
    to functions that may consume significant memory.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before function
        gc.collect()

        # Monitor memory usage
        with memory_profiler(func.__name__):
            result = func(*args, **kwargs)

        # Force garbage collection after function
        gc.collect()

        return result

    return wrapper


class MemoryPool:
    """
    Memory pool for reusing numpy arrays to reduce allocation overhead.

    This class manages a pool of pre-allocated numpy arrays that can be
    reused to avoid frequent memory allocation/deallocation cycles.
    """

    def __init__(self, max_pool_size: int = 100):
        """
        Initialize memory pool.

        Args:
            max_pool_size: Maximum number of arrays to keep in pool
        """
        self.max_pool_size = max_pool_size
        self.pools = {}  # (shape, dtype) -> list of arrays
        self.lock = threading.Lock()

    def get_array(self, shape: Tuple[int, ...], dtype=np.uint8) -> np.ndarray:
        """
        Get an array from the pool or create a new one.

        Args:
            shape: Shape of the array
            dtype: Data type of the array

        Returns:
            Numpy array with specified shape and dtype
        """
        key = (shape, dtype)

        with self.lock:
            if key in self.pools and self.pools[key]:
                array = self.pools[key].pop()
                # Clear the array
                array.fill(0)
                return array

        # Create new array if pool is empty
        return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """
        Return an array to the pool for reuse.

        Args:
            array: Array to return to pool
        """
        if array is None:
            return

        key = (array.shape, array.dtype)

        with self.lock:
            if key not in self.pools:
                self.pools[key] = []

            # Only keep arrays if pool isn't full
            if len(self.pools[key]) < self.max_pool_size:
                self.pools[key].append(array)

    def clear_pool(self):
        """Clear all arrays from the pool."""
        with self.lock:
            self.pools.clear()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory pool."""
        with self.lock:
            stats = {
                "total_pools": len(self.pools),
                "total_arrays": sum(len(pool) for pool in self.pools.values()),
                "pool_details": {
                    str(key): len(pool) for key, pool in self.pools.items()
                },
            }
        return stats


def optimize_opencv_memory():
    """
    Optimize OpenCV memory usage settings.

    This function configures OpenCV to use less memory and be more
    efficient in memory allocation.
    """
    try:
        import cv2

        # Set number of threads for OpenCV
        cv2.setNumThreads(2)

        # Disable OpenCV's memory optimization (can cause issues)
        cv2.setUseOptimized(False)

        logger.info("OpenCV memory optimization applied")

    except ImportError:
        logger.warning("OpenCV not available for memory optimization")


def force_garbage_collection():
    """
    Force comprehensive garbage collection.

    This function performs multiple garbage collection passes to ensure
    maximum memory cleanup.
    """
    # Multiple GC passes for thorough cleanup
    for _ in range(3):
        collected = gc.collect()

    logger.debug(f"Garbage collection completed, collected {collected} objects")


def get_memory_summary() -> Dict[str, Any]:
    """
    Get comprehensive memory usage summary.

    Returns:
        Dictionary containing detailed memory information
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    # System memory
    system_memory = psutil.virtual_memory()

    # Garbage collection stats
    gc_stats = gc.get_stats()

    summary = {
        "process_memory_mb": memory_info.rss / 1024 / 1024,
        "process_memory_percent": process.memory_percent(),
        "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
        "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024,
        "system_memory_percent": system_memory.percent,
        "gc_collections": sum(stat["collections"] for stat in gc_stats),
        "gc_collected": sum(stat["collected"] for stat in gc_stats),
        "gc_uncollectable": sum(stat["uncollectable"] for stat in gc_stats),
    }

    return summary


# Global memory pool instance
_global_memory_pool = None


def get_global_memory_pool() -> MemoryPool:
    """Get the global memory pool instance."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = MemoryPool()
    return _global_memory_pool


# Initialize OpenCV optimization on import
optimize_opencv_memory()
