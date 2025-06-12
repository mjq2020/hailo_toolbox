#!/usr/bin/env python3
"""
Memory monitoring script for Hailo Toolbox inference processes.

This script provides real-time memory monitoring and analysis for debugging
memory leaks and optimizing memory usage in video inference pipelines.

Usage:
    python memory_monitor.py --pid <process_id>
    python memory_monitor.py --command "python -m hailo_toolbox.cli.infer model.onnx --source-type webcam"
"""

import argparse
import time
import psutil
import os
import sys
import subprocess
import threading
import signal
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


class MemoryMonitor:
    """
    Advanced memory monitoring utility for tracking memory usage patterns.
    """

    def __init__(
        self,
        pid: Optional[int] = None,
        command: Optional[str] = None,
        interval: float = 1.0,
        duration: Optional[float] = None,
        threshold_mb: float = 1000.0,
        output_file: Optional[str] = None,
    ):
        """
        Initialize memory monitor.

        Args:
            pid: Process ID to monitor (if None, will launch command)
            command: Command to launch and monitor
            interval: Monitoring interval in seconds
            duration: Maximum monitoring duration in seconds
            threshold_mb: Memory threshold for warnings
            output_file: File to save monitoring results
        """
        self.pid = pid
        self.command = command
        self.interval = interval
        self.duration = duration
        self.threshold_mb = threshold_mb
        self.output_file = output_file

        self.process = None
        self.launched_process = None
        self.monitoring = False
        self.start_time = None

        # Data storage
        self.timestamps = []
        self.memory_usage = []
        self.memory_percent = []
        self.cpu_percent = []
        self.num_threads = []
        self.gc_stats = []

        # Statistics
        self.peak_memory = 0.0
        self.avg_memory = 0.0
        self.memory_growth_rate = 0.0
        self.warnings_count = 0

    def start_monitoring(self) -> bool:
        """
        Start memory monitoring.

        Returns:
            True if monitoring started successfully
        """
        try:
            # Get or launch process
            if self.pid:
                self.process = psutil.Process(self.pid)
                print(f"Monitoring existing process PID: {self.pid}")
            elif self.command:
                print(f"Launching command: {self.command}")
                self.launched_process = subprocess.Popen(
                    self.command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                self.process = psutil.Process(self.launched_process.pid)
                print(f"Monitoring launched process PID: {self.launched_process.pid}")
            else:
                print("Error: Either PID or command must be provided")
                return False

            # Verify process exists
            if not self.process.is_running():
                print("Error: Process is not running")
                return False

            print(f"Process info: {self.process.name()} (PID: {self.process.pid})")
            print(f"Command line: {' '.join(self.process.cmdline())}")

            # Start monitoring
            self.monitoring = True
            self.start_time = time.time()

            # Set up signal handler for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)

            print(
                f"Starting memory monitoring (interval: {self.interval}s, threshold: {self.threshold_mb}MB)"
            )
            print("Press Ctrl+C to stop monitoring and generate report")
            print("-" * 60)

            self._monitor_loop()

            return True

        except psutil.NoSuchProcess:
            print(f"Error: Process with PID {self.pid} not found")
            return False
        except Exception as e:
            print(f"Error starting monitoring: {e}")
            return False

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal for graceful shutdown."""
        print("\nReceived interrupt signal, stopping monitoring...")
        self.stop_monitoring()

    def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring and self.process.is_running():
                # Check duration limit
                if self.duration and (time.time() - self.start_time) > self.duration:
                    print(f"\nReached maximum monitoring duration ({self.duration}s)")
                    break

                # Collect metrics
                try:
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    memory_pct = self.process.memory_percent()
                    cpu_pct = self.process.cpu_percent()
                    threads = self.process.num_threads()

                    # Store data
                    current_time = time.time() - self.start_time
                    self.timestamps.append(current_time)
                    self.memory_usage.append(memory_mb)
                    self.memory_percent.append(memory_pct)
                    self.cpu_percent.append(cpu_pct)
                    self.num_threads.append(threads)

                    # Update statistics
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    if len(self.memory_usage) > 1:
                        self.avg_memory = sum(self.memory_usage) / len(
                            self.memory_usage
                        )

                        # Calculate growth rate (MB per minute)
                        if len(self.memory_usage) >= 10:
                            recent_growth = (
                                (self.memory_usage[-1] - self.memory_usage[-10])
                                / (self.timestamps[-1] - self.timestamps[-10])
                                * 60
                            )
                            self.memory_growth_rate = recent_growth

                    # Check threshold
                    if memory_mb > self.threshold_mb:
                        self.warnings_count += 1
                        print(
                            f"⚠️  WARNING: Memory usage ({memory_mb:.1f}MB) exceeds threshold ({self.threshold_mb}MB)"
                        )

                    # Print status
                    print(
                        f"Time: {current_time:6.1f}s | "
                        f"Memory: {memory_mb:7.1f}MB ({memory_pct:5.1f}%) | "
                        f"CPU: {cpu_pct:5.1f}% | "
                        f"Threads: {threads:3d} | "
                        f"Growth: {self.memory_growth_rate:+6.1f}MB/min"
                    )

                except psutil.AccessDenied:
                    print("Warning: Access denied to process metrics")
                except psutil.NoSuchProcess:
                    print("Process terminated")
                    break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop monitoring and generate report."""
        self.monitoring = False

        if self.launched_process and self.launched_process.poll() is None:
            print("Terminating launched process...")
            self.launched_process.terminate()
            try:
                self.launched_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing launched process...")
                self.launched_process.kill()

        self._generate_report()

    def _generate_report(self):
        """Generate monitoring report."""
        if not self.timestamps:
            print("No data collected")
            return

        print("\n" + "=" * 60)
        print("MEMORY MONITORING REPORT")
        print("=" * 60)

        # Basic statistics
        total_time = self.timestamps[-1]
        print(f"Monitoring Duration: {total_time:.1f} seconds")
        print(f"Data Points: {len(self.timestamps)}")
        print(f"Process PID: {self.process.pid if self.process else 'N/A'}")

        # Memory statistics
        print(f"\nMemory Statistics:")
        print(f"  Peak Memory: {self.peak_memory:.1f} MB")
        print(f"  Average Memory: {self.avg_memory:.1f} MB")
        print(f"  Final Memory: {self.memory_usage[-1]:.1f} MB")
        print(f"  Memory Growth Rate: {self.memory_growth_rate:+.1f} MB/min")
        print(f"  Threshold Violations: {self.warnings_count}")

        # CPU and thread statistics
        if self.cpu_percent:
            avg_cpu = sum(self.cpu_percent) / len(self.cpu_percent)
            max_cpu = max(self.cpu_percent)
            print(f"\nCPU Statistics:")
            print(f"  Average CPU: {avg_cpu:.1f}%")
            print(f"  Peak CPU: {max_cpu:.1f}%")

        if self.num_threads:
            avg_threads = sum(self.num_threads) / len(self.num_threads)
            max_threads = max(self.num_threads)
            print(f"\nThread Statistics:")
            print(f"  Average Threads: {avg_threads:.1f}")
            print(f"  Peak Threads: {max_threads}")

        # Memory leak detection
        self._detect_memory_leaks()

        # Save data if requested
        if self.output_file:
            self._save_data()

        # Generate plots
        self._generate_plots()

    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        print(f"\nMemory Leak Analysis:")

        if len(self.memory_usage) < 10:
            print("  Insufficient data for leak detection")
            return

        # Calculate trend over time
        x = np.array(self.timestamps)
        y = np.array(self.memory_usage)

        # Linear regression to detect trend
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # MB per second
        slope_per_minute = slope * 60

        # Leak detection thresholds
        if slope_per_minute > 10:
            print(
                f"  🚨 SEVERE MEMORY LEAK DETECTED: {slope_per_minute:.1f} MB/min growth"
            )
        elif slope_per_minute > 5:
            print(f"  ⚠️  MODERATE MEMORY LEAK: {slope_per_minute:.1f} MB/min growth")
        elif slope_per_minute > 1:
            print(f"  ⚠️  MINOR MEMORY LEAK: {slope_per_minute:.1f} MB/min growth")
        else:
            print(
                f"  ✅ No significant memory leak detected ({slope_per_minute:.1f} MB/min)"
            )

        # Check for memory spikes
        if len(self.memory_usage) > 5:
            memory_std = np.std(self.memory_usage)
            memory_mean = np.mean(self.memory_usage)

            spikes = [m for m in self.memory_usage if m > memory_mean + 2 * memory_std]
            if spikes:
                print(f"  📈 Memory spikes detected: {len(spikes)} instances")
                print(f"     Largest spike: {max(spikes):.1f} MB")

    def _save_data(self):
        """Save monitoring data to file."""
        try:
            import json

            data = {
                "metadata": {
                    "pid": self.process.pid if self.process else None,
                    "command": self.command,
                    "start_time": datetime.now().isoformat(),
                    "duration": self.timestamps[-1] if self.timestamps else 0,
                    "interval": self.interval,
                    "threshold_mb": self.threshold_mb,
                },
                "statistics": {
                    "peak_memory_mb": self.peak_memory,
                    "avg_memory_mb": self.avg_memory,
                    "memory_growth_rate_mb_per_min": self.memory_growth_rate,
                    "warnings_count": self.warnings_count,
                },
                "data": {
                    "timestamps": self.timestamps,
                    "memory_usage_mb": self.memory_usage,
                    "memory_percent": self.memory_percent,
                    "cpu_percent": self.cpu_percent,
                    "num_threads": self.num_threads,
                },
            }

            with open(self.output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\nData saved to: {self.output_file}")

        except Exception as e:
            print(f"Error saving data: {e}")

    def _generate_plots(self):
        """Generate monitoring plots."""
        try:
            if len(self.timestamps) < 2:
                return

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Memory Monitoring Report", fontsize=16)

            # Memory usage over time
            ax1.plot(self.timestamps, self.memory_usage, "b-", linewidth=2)
            ax1.axhline(
                y=self.threshold_mb,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({self.threshold_mb}MB)",
            )
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Memory Usage (MB)")
            ax1.set_title("Memory Usage Over Time")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Memory percentage
            ax2.plot(self.timestamps, self.memory_percent, "g-", linewidth=2)
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Memory Percentage (%)")
            ax2.set_title("Memory Percentage Over Time")
            ax2.grid(True, alpha=0.3)

            # CPU usage
            if self.cpu_percent:
                ax3.plot(self.timestamps, self.cpu_percent, "r-", linewidth=2)
                ax3.set_xlabel("Time (seconds)")
                ax3.set_ylabel("CPU Usage (%)")
                ax3.set_title("CPU Usage Over Time")
                ax3.grid(True, alpha=0.3)

            # Thread count
            if self.num_threads:
                ax4.plot(self.timestamps, self.num_threads, "purple", linewidth=2)
                ax4.set_xlabel("Time (seconds)")
                ax4.set_ylabel("Number of Threads")
                ax4.set_title("Thread Count Over Time")
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_filename = (
                f"memory_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {plot_filename}")

            # Show plot if possible
            try:
                plt.show()
            except:
                print("Cannot display plot (no GUI available)")

        except ImportError:
            print("Matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Memory monitoring for Hailo Toolbox")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pid", type=int, help="Process ID to monitor")
    group.add_argument("--command", type=str, help="Command to launch and monitor")

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Monitoring interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration", type=float, help="Maximum monitoring duration in seconds"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1000.0,
        help="Memory threshold in MB for warnings (default: 1000)",
    )
    parser.add_argument(
        "--output", type=str, help="Output file to save monitoring data (JSON format)"
    )

    args = parser.parse_args()

    # Create and start monitor
    monitor = MemoryMonitor(
        pid=args.pid,
        command=args.command,
        interval=args.interval,
        duration=args.duration,
        threshold_mb=args.threshold,
        output_file=args.output,
    )

    success = monitor.start_monitoring()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
