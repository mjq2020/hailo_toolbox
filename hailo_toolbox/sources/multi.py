"""
Multi-source manager for handling multiple video sources simultaneously.
"""

import threading
import time
import queue
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union, Set
from .base import BaseSource, SourceType


class MultiSourceManager(BaseSource):
    """
    Manages multiple video sources simultaneously.

    Provides synchronized access to frames from multiple sources using
    a thread pool and frame buffers.
    """

    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a multi-source manager.

        Args:
            source_id: Unique identifier for this source manager.
            config: Configuration dictionary containing:
                - sources: List of source objects or configurations.
                - sync_mode: Synchronization mode ("latest", "nearest", "wait_all").
                - max_queue_size: Maximum queue size per source (default: 30).
                - timeout: Timeout for operations in seconds (default: 10).
                - fps: Target FPS for synchronized output (default: 30).
        """
        super().__init__(source_id, config)
        self.source_type = SourceType.MULTI

        # Multi-source specific configs
        self.sync_mode = self.config.get("sync_mode", "latest")
        self.max_queue_size = self.config.get("max_queue_size", 30)
        self.timeout = self.config.get("timeout", 10)

        # Setup sources
        self.sources = []
        self.source_configs = self.config.get("sources", [])
        self._setup_sources()

        # Thread and queue management
        self.threads = {}
        self.frame_queues = {}
        self.stop_event = threading.Event()
        self.latest_frames = {}

    def _setup_sources(self):
        """
        Set up the individual video sources.
        """
        from .base import BaseSource

        for idx, source_config in enumerate(self.source_configs):
            if isinstance(source_config, BaseSource):
                # Already a source object
                source = source_config
            else:
                # Configuration dictionary for a source
                source_type = source_config.get("type")
                source_id = source_config.get("id", f"{self.source_id}_sub{idx}")

                if isinstance(source_type, str):
                    # Convert string to enum
                    source_type = getattr(SourceType, source_type.upper(), None)

                if source_type is None:
                    raise ValueError(f"Source type must be specified for source {idx}")

                # Create the source
                source = BaseSource.create_source(source_type, source_id, source_config)

            self.sources.append(source)

    def open(self) -> bool:
        """
        Open all sources and start capture threads.

        Returns:
            True if all sources opened successfully, False otherwise.
        """
        try:
            # Open all sources
            open_sources = []
            for source in self.sources:
                if source.open():
                    open_sources.append(source)
                else:
                    # If any source fails to open, close the ones that did open
                    for opened_source in open_sources:
                        opened_source.close()
                    return False

            # Initialize queues
            self.frame_queues = {
                source.source_id: queue.Queue(maxsize=self.max_queue_size)
                for source in self.sources
            }

            # Clear stop event
            self.stop_event.clear()

            # Start capture threads
            self.threads = {}
            for source in self.sources:
                thread = threading.Thread(
                    target=self._capture_thread, args=(source,), daemon=True
                )
                self.threads[source.source_id] = thread
                thread.start()

            self.is_opened = True
            return True

        except Exception as e:
            print(f"Error opening multi-source manager {self.source_id}: {e}")
            self.close()
            return False

    def _capture_thread(self, source: BaseSource) -> None:
        """
        Thread function for capturing frames from a source.

        Args:
            source: Source to capture from.
        """
        source_queue = self.frame_queues[source.source_id]

        while not self.stop_event.is_set():
            try:
                # Read a frame
                success, frame = source.read()
                if not success:
                    # Handle source failure
                    if source.is_open():
                        # Source is still open but read failed, retry
                        time.sleep(0.1)
                        continue
                    else:
                        # Source is closed, try to reopen
                        if not source.open():
                            # Failed to reopen, exit thread
                            break
                        continue

                # Get timestamp
                timestamp = time.time()

                # Add to queue, replacing oldest frame if full
                if source_queue.full():
                    try:
                        source_queue.get_nowait()
                    except queue.Empty:
                        pass

                source_queue.put((timestamp, frame))

                # Also update latest frame
                self.latest_frames[source.source_id] = (timestamp, frame)

                # Control capture rate based on the target FPS
                time.sleep(1.0 / (self.fps * 2))  # Capture at 2x the target rate

            except Exception as e:
                print(f"Error in capture thread for source {source.source_id}: {e}")
                time.sleep(0.5)  # Brief delay before retry

    def read(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        Read frames from all sources based on the synchronization mode.

        Returns:
            Tuple of (success, frames) where frames is a dictionary mapping
            source IDs to frames.
        """
        if not self.is_opened:
            return False, None

        # Different sync strategies
        if self.sync_mode == "latest":
            return self._read_latest()
        elif self.sync_mode == "nearest":
            return self._read_nearest()
        elif self.sync_mode == "wait_all":
            return self._read_wait_all()
        else:
            raise ValueError(f"Unknown sync mode: {self.sync_mode}")

    def _read_latest(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        Read the latest available frame from each source.

        Returns:
            Tuple of (success, frames dictionary).
        """
        frames = {}

        # Check if we have any frames
        if not self.latest_frames:
            return False, None

        # Get latest frames
        for source_id, (timestamp, frame) in self.latest_frames.items():
            frames[source_id] = frame

        return True, frames

    def _read_nearest(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        Read frames from each source that are closest in time to each other.

        Returns:
            Tuple of (success, frames dictionary).
        """
        frames = {}
        all_timestamps = []

        # Get all available frame timestamps
        for source_id, source_queue in self.frame_queues.items():
            if source_queue.empty():
                continue

            queue_list = list(source_queue.queue)
            for timestamp, _ in queue_list:
                all_timestamps.append((timestamp, source_id))

        if not all_timestamps:
            return False, None

        # Sort by timestamp
        all_timestamps.sort()

        # Calculate median timestamp
        median_timestamp = all_timestamps[len(all_timestamps) // 2][0]

        # Get frames closest to the median timestamp
        for source_id, source_queue in self.frame_queues.items():
            if source_queue.empty():
                continue

            closest_frame = None
            closest_delta = float("inf")

            queue_list = list(source_queue.queue)
            for timestamp, frame in queue_list:
                delta = abs(timestamp - median_timestamp)
                if delta < closest_delta:
                    closest_delta = delta
                    closest_frame = frame

            if closest_frame is not None:
                frames[source_id] = closest_frame

        # Only return if we have frames from all sources
        if len(frames) == len(self.sources):
            return True, frames
        else:
            return False, None

    def _read_wait_all(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        Wait until all sources have at least one frame available.

        Returns:
            Tuple of (success, frames dictionary).
        """
        frames = {}

        # Check if all queues have frames
        all_have_frames = all(not q.empty() for q in self.frame_queues.values())

        if not all_have_frames:
            # Wait a bit for frames to arrive
            time.sleep(0.1)
            all_have_frames = all(not q.empty() for q in self.frame_queues.values())

            if not all_have_frames:
                return False, None

        # Get newest frame from each queue
        for source_id, source_queue in self.frame_queues.items():
            if not source_queue.empty():
                timestamp, frame = source_queue.get()
                frames[source_id] = frame

        return True, frames

    def close(self) -> None:
        """Close all sources and stop capture threads."""
        # Signal threads to stop
        self.stop_event.set()

        # Wait for threads to finish
        for thread_id, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)

        # Close all sources
        for source in self.sources:
            if source.is_open():
                source.close()

        # Clear queues
        for queue in self.frame_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass

        self.latest_frames.clear()
        self.threads.clear()
        self.frame_queues.clear()
        self.is_opened = False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the multi-source manager.

        Returns:
            Dictionary with information about the manager and its sources.
        """
        info = super().get_info()

        # Add source information
        sources_info = []
        for source in self.sources:
            sources_info.append(
                {
                    "id": source.source_id,
                    "type": source.source_type,
                    "is_open": source.is_open(),
                    "queue_size": (
                        len(self.frame_queues[source.source_id].queue)
                        if source.source_id in self.frame_queues
                        else 0
                    ),
                }
            )

        info.update(
            {
                "sync_mode": self.sync_mode,
                "num_sources": len(self.sources),
                "sources": sources_info,
                "active_threads": sum(1 for t in self.threads.values() if t.is_alive()),
            }
        )

        return info
