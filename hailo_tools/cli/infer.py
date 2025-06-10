"""
Command-line interface for model inference.
"""

import os
import sys
import argparse
import yaml
import json
import importlib
import inspect
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from ..sources import (
    BaseSource, SourceType, 
    FileSource, WebcamSource, IPCameraSource, MultiSourceManager
)
from ..inference import (
    BaseInferenceEngine, InferenceResult, InferenceCallback,
    ONNXInferenceEngine, InferencePipeline
)
from ..utils import setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def get_source(source_type: str, source_id: str, config: Dict[str, Any]) -> BaseSource:
    """
    Create a source instance based on the source type.
    
    Args:
        source_type: Type of source ("file", "webcam", "ip_camera", "multi").
        source_id: Identifier for the source.
        config: Source configuration.
        
    Returns:
        Configured source.
    """
    source_type = source_type.lower()
    
    if source_type == "file":
        return FileSource(source_id, config)
    elif source_type in ["webcam", "usb"]:
        return WebcamSource(source_id, config)
    elif source_type in ["ip", "ip_camera", "ipcamera", "rtsp"]:
        return IPCameraSource(source_id, config)
    elif source_type == "multi":
        return MultiSourceManager(source_id, config)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def get_engine(model_path: str, config: Dict[str, Any]) -> BaseInferenceEngine:
    """
    Create an inference engine for the model.
    
    Args:
        model_path: Path to the model file.
        config: Engine configuration.
        
    Returns:
        Configured inference engine.
    """
    # Check model file extension
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == ".onnx":
        return ONNXInferenceEngine(model_path, config)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def load_callback_function(callback_path: str) -> InferenceCallback:
    """
    Load a callback function from a Python module.
    
    Args:
        callback_path: Path to the callback in the format "module.submodule:function".
        
    Returns:
        Loaded callback function.
    """
    try:
        module_path, function_name = callback_path.split(":")
        module = importlib.import_module(module_path)
        callback = getattr(module, function_name)
        
        # Verify it's callable
        if not callable(callback):
            raise ValueError(f"{callback_path} is not callable")
            
        # Verify it has the correct signature
        sig = inspect.signature(callback)
        if len(sig.parameters) < 1:
            raise ValueError(f"{callback_path} must accept at least one parameter (InferenceResult)")
            
        return callback
    except Exception as e:
        raise ImportError(f"Failed to load callback function {callback_path}: {e}")


def default_callback(result: InferenceResult) -> None:
    """
    Default callback function for inference results.
    
    Args:
        result: Inference result to process.
    """
    if not result.success:
        print(f"Inference failed: {result.metadata.get('error', 'Unknown error')}")
        return
        
    # Print basic information
    print(f"Model: {result.model_name}")
    print(f"Inference time: {result.inference_time_ms:.2f} ms")
    
    # If it's an image, show it
    if result.input_data:
        first_input = next(iter(result.input_data.values()))
        
        # Check if the first input is an image-like array
        if isinstance(first_input, np.ndarray) and len(first_input.shape) >= 3:
            # Get first batch element if there's a batch dimension
            if len(first_input.shape) == 4:
                img = first_input[0]
                if img.shape[0] in [1, 3]:  # CHW format
                    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            else:
                img = first_input
                
            # Ensure it's suitable for display
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
                
            # Display the image with OpenCV
            cv2.imshow("Input", img)
            cv2.waitKey(1)


def visualize_callback(result: InferenceResult) -> None:
    """
    Visualization callback function for common model types.
    
    This function attempts to visualize the outputs of common model types
    like classification, object detection, segmentation, etc.
    
    Args:
        result: Inference result to visualize.
    """
    if not result.success:
        print(f"Inference failed: {result.metadata.get('error', 'Unknown error')}")
        return
        
    # Get the first input image if available
    input_img = None
    if result.input_data:
        first_input = next(iter(result.input_data.values()))
        
        # Check if the first input is an image-like array
        if isinstance(first_input, np.ndarray) and len(first_input.shape) >= 3:
            # Get first batch element if there's a batch dimension
            if len(first_input.shape) == 4:
                img = first_input[0]
                if img.shape[0] in [1, 3]:  # CHW format
                    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            else:
                img = first_input
                
            # Ensure it's suitable for display
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                
            # Make a copy for visualization
            input_img = img.copy()
    
    # Simple attempt at auto-detecting model type and visualizing
    if input_img is not None:
        # Find the shape of the first raw output
        first_output_name = next(iter(result.raw_outputs.keys()))
        first_output = result.raw_outputs[first_output_name]
        
        # Object detection (assume output is bounding boxes + class scores)
        if len(first_output.shape) == 2 and first_output.shape[1] > 4:
            # Assume format is [batch, num_detections, 5+] where 5+ is [x, y, w, h, score, class_id, ...]
            # Extract first batch
            if len(first_output.shape) > 2:
                detections = first_output[0]
            else:
                detections = first_output
                
            # Draw boxes
            vis_img = input_img.copy()
            h, w = vis_img.shape[:2]
            
            for det in detections:
                if len(det) >= 5:  # Ensure we have at least x,y,w,h,score
                    # Extract normalized coordinates (assuming they're normalized)
                    score = det[4]
                    if score > 0.5:  # Confidence threshold
                        # Different formats handle coordinates differently
                        x1, y1, x2, y2 = None, None, None, None
                        
                        # Try common formats
                        if det[0] < 1.0 and det[1] < 1.0 and det[2] < 1.0 and det[3] < 1.0:
                            # Normalized [x1, y1, x2, y2] or [x, y, w, h]
                            if det[2] <= 1.0 and det[3] <= 1.0:
                                # Could be [x1, y1, w, h] or [x, y, x2, y2]
                                if det[0] + det[2] <= 1.0 and det[1] + det[3] <= 1.0:
                                    # [x, y, w, h]
                                    x1, y1 = int(det[0] * w), int(det[1] * h)
                                    x2, y2 = int((det[0] + det[2]) * w), int((det[1] + det[3]) * h)
                                else:
                                    # [x1, y1, x2, y2]
                                    x1, y1 = int(det[0] * w), int(det[1] * h)
                                    x2, y2 = int(det[2] * w), int(det[3] * h)
                        else:
                            # Assuming raw pixel coordinates
                            x1, y1 = int(det[0]), int(det[1])
                            x2, y2 = int(det[2]), int(det[3])
                            
                        # Draw the box
                        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add score label
                            label = f"{score:.2f}"
                            if len(det) >= 6:  # If class ID is available
                                label = f"Class {int(det[5])}: {score:.2f}"
                                
                            cv2.putText(vis_img, label, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the image
            cv2.imshow("Detections", vis_img)
            cv2.waitKey(1)
            
        # Image classification
        elif len(first_output.shape) == 2 and first_output.shape[1] <= 1000:
            # Assume format is [batch, num_classes]
            # Extract first batch
            if len(first_output.shape) > 1:
                scores = first_output[0]
            else:
                scores = first_output
                
            # Get top 5 classes
            top_indices = np.argsort(scores)[-5:][::-1]
            top_scores = scores[top_indices]
            
            # Display image with classification results
            vis_img = input_img.copy()
            h, w = vis_img.shape[:2]
            
            # Add black bar at the bottom for text
            bar_height = 100
            result_img = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
            result_img[:h, :] = vis_img
            
            # Add classification results
            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                label = f"Class {idx}: {score:.4f}"
                cv2.putText(result_img, label, (10, h + 20 + i*15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            # Show the image
            cv2.imshow("Classification", result_img)
            cv2.waitKey(1)
            
        # Segmentation
        elif len(first_output.shape) >= 3:
            # Assume format is [batch, num_classes, height, width] or [batch, height, width]
            # Extract first batch
            if len(first_output.shape) >= 4:
                # Multi-class segmentation
                seg_map = np.argmax(first_output[0], axis=0)
            else:
                # Binary segmentation
                seg_map = first_output[0] > 0.5
                
            # Resize to match input image
            h, w = input_img.shape[:2]
            if seg_map.shape[0] != h or seg_map.shape[1] != w:
                seg_map = cv2.resize(seg_map.astype(np.float32), (w, h), 
                                      interpolation=cv2.INTER_NEAREST)
                
            # Colorize segmentation map
            color_map = np.zeros((h, w, 3), dtype=np.uint8)
            num_classes = np.max(seg_map) + 1
            
            # Generate random colors for each class
            colors = np.random.randint(0, 255, size=(int(num_classes), 3), dtype=np.uint8)
            
            # Apply colors to segmentation map
            for class_idx in range(int(num_classes)):
                mask = seg_map == class_idx
                color_map[mask] = colors[class_idx]
                
            # Create a blended output
            alpha = 0.5
            blended = cv2.addWeighted(input_img, 1 - alpha, color_map, alpha, 0)
            
            # Show the segmentation
            cv2.imshow("Segmentation", blended)
            cv2.waitKey(1)
            
        else:
            # Unknown model type, just show the input
            cv2.imshow("Input", input_img)
            cv2.waitKey(1)
    
    # Print inference information
    total_time = result.preprocessing_time_ms + result.inference_time_ms + result.postprocessing_time_ms
    print(f"Total time: {total_time:.2f} ms (pre: {result.preprocessing_time_ms:.2f}, infer: {result.inference_time_ms:.2f}, post: {result.postprocessing_time_ms:.2f})")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with deep learning models")
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model file (ONNX format)"
    )
    
    parser.add_argument(
        "--source-type",
        type=str,
        default="webcam",
        choices=["file", "webcam", "ip_camera", "multi"],
        help="Type of video source"
    )
    
    parser.add_argument(
        "--source-path",
        type=str,
        help="Path to the source (file path, camera URL, etc.)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML or JSON configuration file"
    )
    
    parser.add_argument(
        "--callback",
        type=str,
        help="Python callback function for inference results (module.submodule:function)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable basic visualization of results"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="sync",
        choices=["sync", "async", "batch"],
        help="Inference mode"
    )
    
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the source if it's a file"
    )
    
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS information"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="infer.log",
        help="Log file name"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the model inference CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(
        name="hailo_tools.infer", 
        level=log_level,
        log_file=args.log_file,
        log_dir=args.log_dir
    )
    
    try:
        # Load config from file if specified
        config = {}
        if args.config:
            config = load_config(args.config)
            logger.info("已加载配置文件: %s", args.config)
            
        # Set up the source
        source_config = config.get("source", {})
        
        # Override with command-line arguments
        if args.source_path:
            if args.source_type == "file":
                source_config["file_path"] = args.source_path
            elif args.source_type == "ip_camera":
                source_config["url"] = args.source_path
            elif args.source_type == "webcam" and args.source_path.isdigit():
                source_config["device_id"] = int(args.source_path)
                
        # Set up the inference engine
        engine_config = config.get("engine", {})
        
        # Set up the pipeline
        pipeline_config = config.get("pipeline", {})
        
        # Override with command-line arguments
        if args.mode:
            pipeline_config["mode"] = args.mode
        if args.loop:
            pipeline_config["loop"] = True
        if args.show_fps:
            pipeline_config["show_fps"] = True
            
        # Create the source
        try:
            source = get_source(args.source_type, "main_source", source_config)
            logger.info("创建 %s 视频源", args.source_type)
        except Exception as e:
            logger.exception("创建视频源错误")
            return 1
            
        # Create the inference engine
        try:
            engine = get_engine(args.model_path, engine_config)
            logger.info("创建推理引擎: %s", args.model_path)
        except Exception as e:
            logger.exception("创建推理引擎错误")
            return 1
            
        # Set up the callback
        callback = None
        if args.callback:
            try:
                callback = load_callback_function(args.callback)
                logger.info("加载回调函数: %s", args.callback)
            except Exception as e:
                logger.exception("加载回调函数错误")
                return 1
        elif args.visualize:
            callback = visualize_callback
            logger.info("使用内置可视化回调")
        else:
            callback = default_callback
            logger.info("使用默认回调")
            
        # Set the callback
        pipeline_config["infer_callback"] = callback
        
        # Create and run the pipeline
        try:
            pipeline = InferencePipeline(pipeline_config)
            pipeline.set_source(source)
            pipeline.set_engine(engine)
            
            logger.info("启动推理管道")
            pipeline.run()
            
            # Display final statistics
            stats = pipeline.get_stats()
            logger.info("已处理 %d 帧，平均 %.2f FPS", stats['frame_count'], stats['avg_fps'])
            logger.info("平均处理时间: %.2f ms", stats['avg_processing_time_ms'])
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("用户中断推理")
            return 0
            
        except Exception as e:
            logger.exception("推理管道错误")
            return 1
            
    except Exception as e:
        logger.exception("意外错误")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 