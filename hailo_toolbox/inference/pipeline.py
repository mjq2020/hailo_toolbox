"""
Pipeline for running inference with video sources and models.
"""

import time
import threading
import queue
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import numpy as np
from ..sources import BaseSource
from .base import BaseInferenceEngine, InferenceResult, InferenceCallback


class InferencePipeline:
    """
    A pipeline that connects video sources with inference engines.

    The pipeline can operate in different modes:
    - Synchronous: Process frames one by one
    - Asynchronous: Process frames in a separate thread
    - Batched: Process multiple frames at once
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the inference pipeline.

        Args:
            config: Configuration dictionary containing:
                - mode: Pipeline mode ("sync", "async", "batch") (default: "sync").
                - max_queue_size: Maximum queue size for async mode (default: 30).
                - batch_size: Batch size for batch mode (default: 4).
                - skip_frames: Number of frames to skip between inferences (default: 0).
                - show_fps: Whether to print FPS information (default: False).
                - loop: Whether to loop in run_once mode (default: False).
                - warmup_iters: Number of warmup iterations (default: 2).
                - infer_callback: Callback function for inference results.
        """
        self.config = config or {}

        # Pipeline settings
        self.mode = self.config.get("mode", "sync")
        self.max_queue_size = self.config.get("max_queue_size", 30)
        self.batch_size = self.config.get("batch_size", 4)
        self.skip_frames = self.config.get("skip_frames", 0)
        self.show_fps = self.config.get("show_fps", False)
        self.loop = self.config.get("loop", False)
        self.warmup_iters = self.config.get("warmup_iters", 2)

        # Components
        self.source = None
        self.engine = None
        self.infer_callback = self.config.get("infer_callback")

        # Pipeline state
        self.frame_count = 0
        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=self.max_queue_size)
        self.result_queue = queue.Queue(maxsize=self.max_queue_size)

        # Threading
        self.process_thread = None
        self.visualize_thread = None

        # Timing and profiling
        self.start_time = None
        self.fps_history = []
        self.processing_times = []
        self.last_fps_report_time = 0
        self.fps_report_interval = 5.0  # in seconds

    def set_source(self, source: BaseSource) -> None:
        """
        Set the video source for the pipeline.

        Args:
            source: Video source to use.
        """
        self.source = source

    def set_engine(self, engine: BaseInferenceEngine) -> None:
        """
        Set the inference engine for the pipeline.

        Args:
            engine: Inference engine to use.
        """
        self.engine = engine

        # If engine has a callback and we don't, use the engine's callback
        if self.infer_callback is None and engine.callback is not None:
            self.infer_callback = engine.callback
        # If we have a callback and engine doesn't, set engine's callback to ours
        elif self.infer_callback is not None and engine.callback is None:
            engine.set_callback(self.infer_callback)

    def set_callback(self, callback: InferenceCallback) -> None:
        """
        Set a callback function for inference results.

        Args:
            callback: Callback function to use.
        """
        self.infer_callback = callback

        # Also set the callback for the engine if it exists
        if self.engine is not None:
            self.engine.set_callback(callback)

    def start(self) -> bool:
        """
        Start the inference pipeline.

        Returns:
            True if the pipeline was started successfully, False otherwise.
        """
        if self.source is None:
            print("No source set")
            return False

        if self.engine is None:
            print("No inference engine set")
            return False

        # Open source
        if not self.source.is_open() and not self.source.open():
            print("Failed to open source")
            return False

        # Load engine
        if not self.engine.is_model_loaded() and not self.engine.load():
            print("Failed to load model")
            return False

        # Reset state
        self.frame_count = 0
        self.stopped = False
        self.fps_history.clear()
        self.processing_times.clear()

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        # Start appropriate processing based on mode
        if self.mode == "async":
            # Start processing thread
            self.process_thread = threading.Thread(
                target=self._process_frames_async, daemon=True
            )
            self.process_thread.start()

            if self.infer_callback:
                # Start visualization thread if callback exists
                self.visualize_thread = threading.Thread(
                    target=self._process_results, daemon=True
                )
                self.visualize_thread.start()

        elif self.mode == "batch":
            # Start batch processing thread
            self.process_thread = threading.Thread(
                target=self._process_frames_batch, daemon=True
            )
            self.process_thread.start()

            if self.infer_callback:
                # Start visualization thread if callback exists
                self.visualize_thread = threading.Thread(
                    target=self._process_results, daemon=True
                )
                self.visualize_thread.start()

        # Record start time for FPS calculation
        self.start_time = time.time()
        self.last_fps_report_time = self.start_time

        return True

    def stop(self) -> None:
        """Stop the inference pipeline."""
        self.stopped = True

        # Wait for threads to finish
        if self.process_thread is not None and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)

        if self.visualize_thread is not None and self.visualize_thread.is_alive():
            self.visualize_thread.join(timeout=2.0)

        # Close the source
        if self.source is not None and self.source.is_open():
            self.source.close()

        # Print final statistics
        if self.show_fps and self.frame_count > 0 and self.start_time is not None:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"Average FPS: {avg_fps:.2f}")

            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                print(f"Average processing time: {avg_time:.2f} ms")

    def run_once(self) -> Optional[InferenceResult]:
        """
        Run a single inference cycle.

        Returns:
            InferenceResult if successful, None otherwise.
        """
        if self.source is None or self.engine is None:
            return None

        # Ensure source is open
        if not self.source.is_open() and not self.source.open():
            return None

        # Ensure model is loaded
        if not self.engine.is_model_loaded() and not self.engine.load():
            return None

        # Read a frame
        success, frame = self.source.read()
        if not success:
            if self.loop:
                # Try to reopen the source if looping
                self.source.close()
                if not self.source.open():
                    return None
                success, frame = self.source.read()
                if not success:
                    return None
            else:
                return None

        # Warmup
        if self.frame_count < self.warmup_iters:
            # Perform warmup inferences
            for _ in range(self.warmup_iters):
                self.engine(frame)
            self.frame_count = self.warmup_iters

        # Run inference
        start_time = time.time()
        result = self.engine(frame)
        processing_time = (time.time() - start_time) * 1000

        self.frame_count += 1
        self.processing_times.append(processing_time)

        # Call callback if provided
        if self.infer_callback is not None:
            self.infer_callback(result)

        # Show FPS if requested
        if self.show_fps:
            current_time = time.time()
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)

            # Print FPS periodically
            if current_time - self.last_fps_report_time > self.fps_report_interval:
                avg_fps = (
                    sum(self.fps_history) / len(self.fps_history)
                    if self.fps_history
                    else 0
                )
                avg_time = (
                    sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times
                    else 0
                )
                print(
                    f"FPS: {fps:.2f}, Average: {avg_fps:.2f}, Processing Time: {avg_time:.2f} ms"
                )
                self.last_fps_report_time = current_time

        return result

    def run(self) -> None:
        """
        Run the inference pipeline until stopped.

        For synchronous mode, this processes frames directly in the current thread.
        For async and batch modes, this feeds frames to the processing threads.
        """
        if not self.start():
            return

        try:
            # Main loop
            while not self.stopped:
                if self.mode == "sync":
                    # Synchronous mode - process frames directly
                    result = self.run_once()
                    if result is None:
                        # End of source
                        break
                else:
                    # Async or batch mode - feed frames to the queue
                    success, frame = self.source.read()
                    if not success:
                        if self.loop:
                            # Try to reopen the source if looping
                            self.source.close()
                            if not self.source.open():
                                break
                            continue
                        else:
                            break

                    # Skip frames if requested
                    if self.skip_frames > 0:
                        if self.frame_count % (self.skip_frames + 1) == 0:
                            # Only process every (skip_frames + 1)th frame
                            try:
                                if not self.frame_queue.full():
                                    self.frame_queue.put(
                                        (self.frame_count, frame), block=False
                                    )
                            except:
                                pass
                    else:
                        # Process every frame
                        try:
                            if not self.frame_queue.full():
                                self.frame_queue.put(
                                    (self.frame_count, frame), block=False
                                )
                        except:
                            pass

                    self.frame_count += 1

                    # Avoid spinning too fast when the queue is full
                    if self.frame_queue.full():
                        time.sleep(0.01)

        finally:
            self.stop()

    def _process_frames_async(self) -> None:
        """
        Process frames asynchronously in a separate thread.

        This method is used in async mode.
        """
        # Warmup
        if self.warmup_iters > 0:
            # Get a sample frame
            success, frame = self.source.read()
            if success:
                # Perform warmup inferences
                for _ in range(self.warmup_iters):
                    self.engine(frame)

        while not self.stopped:
            try:
                # Get a frame from the queue
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue

                frame_id, frame = self.frame_queue.get(timeout=0.1)

                # Run inference
                start_time = time.time()
                result = self.engine(frame)
                processing_time = (time.time() - start_time) * 1000

                result.frame_id = frame_id

                # Update profiling info
                self.processing_times.append(processing_time)

                # Put result in output queue for visualization thread
                if not self.result_queue.full():
                    self.result_queue.put(result, block=False)

                # Show FPS if requested
                if self.show_fps:
                    current_time = time.time()
                    elapsed = current_time - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.fps_history.append(fps)

                    # Print FPS periodically
                    if (
                        current_time - self.last_fps_report_time
                        > self.fps_report_interval
                    ):
                        avg_fps = (
                            sum(self.fps_history) / len(self.fps_history)
                            if self.fps_history
                            else 0
                        )
                        avg_time = (
                            sum(self.processing_times) / len(self.processing_times)
                            if self.processing_times
                            else 0
                        )
                        print(
                            f"FPS: {fps:.2f}, Average: {avg_fps:.2f}, Processing Time: {avg_time:.2f} ms"
                        )
                        self.last_fps_report_time = current_time

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in process_frames_async: {e}")

    def _process_frames_batch(self) -> None:
        """
        Process frames in batches in a separate thread.

        This method is used in batch mode.
        """
        # Warmup
        if self.warmup_iters > 0:
            # Get a sample frame
            success, frame = self.source.read()
            if success:
                # Perform warmup inferences
                for _ in range(self.warmup_iters):
                    self.engine(frame)

        while not self.stopped:
            try:
                # Collect a batch of frames
                batch_frames = []
                batch_ids = []

                # Wait for enough frames for a batch or until timeout
                start_wait = time.time()
                while len(batch_frames) < self.batch_size and not self.stopped:
                    if self.frame_queue.empty():
                        # Break if we've been waiting too long
                        if time.time() - start_wait > 1.0 and batch_frames:
                            break
                        time.sleep(0.01)
                        continue

                    frame_id, frame = self.frame_queue.get(timeout=0.1)
                    batch_frames.append(frame)
                    batch_ids.append(frame_id)

                    # Reset wait timer
                    start_wait = time.time()

                if not batch_frames:
                    continue

                # Process each frame in the batch sequentially
                # (actual batching would require model-specific implementation)
                for i, frame in enumerate(batch_frames):
                    # Run inference
                    start_time = time.time()
                    result = self.engine(frame)
                    processing_time = (time.time() - start_time) * 1000

                    result.frame_id = batch_ids[i]

                    # Update profiling info
                    self.processing_times.append(processing_time)

                    # Put result in output queue for visualization thread
                    if not self.result_queue.full():
                        self.result_queue.put(result, block=False)

                # Show FPS if requested
                if self.show_fps:
                    current_time = time.time()
                    elapsed = current_time - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.fps_history.append(fps)

                    # Print FPS periodically
                    if (
                        current_time - self.last_fps_report_time
                        > self.fps_report_interval
                    ):
                        avg_fps = (
                            sum(self.fps_history) / len(self.fps_history)
                            if self.fps_history
                            else 0
                        )
                        avg_time = (
                            sum(self.processing_times) / len(self.processing_times)
                            if self.processing_times
                            else 0
                        )
                        print(
                            f"FPS: {fps:.2f}, Average: {avg_fps:.2f}, Processing Time: {avg_time:.2f} ms"
                        )
                        self.last_fps_report_time = current_time

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in process_frames_batch: {e}")

    def _process_results(self) -> None:
        """
        Process inference results in a separate thread.

        This method calls the callback function for each result.
        """
        while not self.stopped:
            try:
                # Get a result from the queue
                if self.result_queue.empty():
                    time.sleep(0.01)
                    continue

                result = self.result_queue.get(timeout=0.1)

                # Call the callback
                if self.infer_callback is not None:
                    self.infer_callback(result)

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in process_results: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with statistics.
        """
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times
            else 0
        )

        return {
            "mode": self.mode,
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "fps": fps,
            "avg_fps": avg_fps,
            "avg_processing_time_ms": avg_processing_time,
            "max_processing_time_ms": (
                max(self.processing_times) if self.processing_times else 0
            ),
            "min_processing_time_ms": (
                min(self.processing_times) if self.processing_times else 0
            ),
            "queue_sizes": {
                "frame_queue": self.frame_queue.qsize(),
                "result_queue": self.result_queue.qsize(),
            },
        }
