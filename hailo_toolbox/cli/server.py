"""
Command-line interface for starting an inference server.
"""

import os
import sys
import argparse
import yaml
import json
import importlib
import time
import threading
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import queue

from flask import Flask, request, jsonify, send_file, Response
import numpy as np
import cv2

from ..sources import (
    BaseSource, SourceType, 
    FileSource, WebcamSource, IPCameraSource, MultiSourceManager
)
from ..inference import (
    BaseInferenceEngine, InferenceResult, InferenceCallback,
    ONNXInferenceEngine, InferencePipeline
)


# Global variables
app = Flask(__name__)
logger = None
engines = {}  # Dictionary of loaded engines
sources = {}  # Dictionary of active sources
pipelines = {}  # Dictionary of active pipelines
result_queues = {}  # Dictionary of result queues
lock = threading.Lock()  # Lock for thread safety


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to use verbose logging.
        
    Returns:
        Configured logger.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger('hailo_toolbox.server')
    logger.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    
    # Also set up Flask logger
    flask_logger = logging.getLogger('werkzeug')
    flask_logger.setLevel(log_level)
    
    return logger


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


def result_callback(pipeline_id: str) -> InferenceCallback:
    """
    Create a callback function for a specific pipeline.
    
    Args:
        pipeline_id: ID of the pipeline.
        
    Returns:
        Callback function.
    """
    def callback(result: InferenceResult) -> None:
        # Store the result in the corresponding queue
        if pipeline_id in result_queues:
            # If the queue is full, remove the oldest result
            if result_queues[pipeline_id].full():
                try:
                    result_queues[pipeline_id].get_nowait()
                except queue.Empty:
                    pass
                    
            # Add the new result
            result_queues[pipeline_id].put(result)
            
    return callback


def start_pipeline(pipeline_id: str, source_id: str, engine_id: str, config: Dict[str, Any]) -> bool:
    """
    Start an inference pipeline.
    
    Args:
        pipeline_id: ID for the pipeline.
        source_id: ID of the source to use.
        engine_id: ID of the engine to use.
        config: Pipeline configuration.
        
    Returns:
        True if the pipeline was started successfully, False otherwise.
    """
    with lock:
        # Check if source and engine exist
        if source_id not in sources:
            logger.error(f"Source {source_id} not found")
            return False
            
        if engine_id not in engines:
            logger.error(f"Engine {engine_id} not found")
            return False
            
        # Create result queue
        max_queue_size = config.get("max_queue_size", 10)
        result_queues[pipeline_id] = queue.Queue(maxsize=max_queue_size)
        
        # Set callback
        config["infer_callback"] = result_callback(pipeline_id)
        
        # Create pipeline
        pipeline = InferencePipeline(config)
        pipeline.set_source(sources[source_id])
        pipeline.set_engine(engines[engine_id])
        
        # Start the pipeline in a separate thread
        thread = threading.Thread(target=pipeline.run, daemon=True)
        thread.start()
        
        # Store the pipeline and thread
        pipelines[pipeline_id] = {
            "pipeline": pipeline,
            "thread": thread,
            "source_id": source_id,
            "engine_id": engine_id,
            "start_time": time.time()
        }
        
        logger.info(f"Started pipeline {pipeline_id} with source {source_id} and engine {engine_id}")
        return True


def stop_pipeline(pipeline_id: str) -> bool:
    """
    Stop an inference pipeline.
    
    Args:
        pipeline_id: ID of the pipeline to stop.
        
    Returns:
        True if the pipeline was stopped successfully, False otherwise.
    """
    with lock:
        if pipeline_id not in pipelines:
            logger.error(f"Pipeline {pipeline_id} not found")
            return False
            
        # Stop the pipeline
        pipeline_info = pipelines[pipeline_id]
        pipeline = pipeline_info["pipeline"]
        pipeline.stop()
        
        # Wait for the thread to finish
        thread = pipeline_info["thread"]
        if thread.is_alive():
            thread.join(timeout=2.0)
            
        # Clean up resources
        if pipeline_id in result_queues:
            # Clear the result queue
            while not result_queues[pipeline_id].empty():
                try:
                    result_queues[pipeline_id].get_nowait()
                except queue.Empty:
                    break
                    
            del result_queues[pipeline_id]
            
        del pipelines[pipeline_id]
        
        logger.info(f"Stopped pipeline {pipeline_id}")
        return True


# API routes

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get a list of all loaded models."""
    model_list = []
    for engine_id, engine in engines.items():
        model_info = engine.get_model_info()
        model_list.append({
            "id": engine_id,
            "name": model_info["model_name"],
            "path": model_info["model_path"],
            "is_loaded": model_info["is_loaded"]
        })
    return jsonify({"models": model_list})


@app.route('/api/models', methods=['POST'])
def load_model():
    """Load a new model."""
    data = request.json
    
    # Required fields
    if 'path' not in data:
        return jsonify({"error": "Model path is required"}), 400
        
    model_path = data['path']
    model_id = data.get('id', str(uuid.uuid4()))
    config = data.get('config', {})
    
    try:
        # Create the engine
        engine = get_engine(model_path, config)
        
        # Load the model
        if not engine.load():
            return jsonify({"error": "Failed to load model"}), 500
            
        # Store the engine
        with lock:
            engines[model_id] = engine
            
        # Return success
        return jsonify({
            "id": model_id,
            "name": engine.model_name,
            "path": model_path,
            "is_loaded": engine.is_loaded
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<model_id>', methods=['DELETE'])
def unload_model(model_id):
    """Unload a model."""
    with lock:
        if model_id not in engines:
            return jsonify({"error": "Model not found"}), 404
            
        # Check if the model is in use by any pipeline
        for pipeline_id, pipeline_info in pipelines.items():
            if pipeline_info["engine_id"] == model_id:
                return jsonify({"error": f"Model is in use by pipeline {pipeline_id}"}), 400
                
        # Unload the model
        engines[model_id].unload()
        del engines[model_id]
        
        return jsonify({"success": True})


@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Get a list of all sources."""
    source_list = []
    for source_id, source in sources.items():
        source_info = source.get_info()
        source_list.append({
            "id": source_id,
            "type": str(source_info["type"]),
            "is_open": source_info["is_open"]
        })
    return jsonify({"sources": source_list})


@app.route('/api/sources', methods=['POST'])
def create_source():
    """Create a new source."""
    data = request.json
    
    # Required fields
    if 'type' not in data:
        return jsonify({"error": "Source type is required"}), 400
        
    source_type = data['type']
    source_id = data.get('id', str(uuid.uuid4()))
    config = data.get('config', {})
    
    try:
        # Create the source
        source = get_source(source_type, source_id, config)
        
        # Store the source
        with lock:
            sources[source_id] = source
            
        # Return success
        return jsonify({
            "id": source_id,
            "type": source_type,
            "is_open": source.is_open()
        })
        
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/sources/<source_id>', methods=['DELETE'])
def delete_source(source_id):
    """Delete a source."""
    with lock:
        if source_id not in sources:
            return jsonify({"error": "Source not found"}), 404
            
        # Check if the source is in use by any pipeline
        for pipeline_id, pipeline_info in pipelines.items():
            if pipeline_info["source_id"] == source_id:
                return jsonify({"error": f"Source is in use by pipeline {pipeline_id}"}), 400
                
        # Close the source
        source = sources[source_id]
        if source.is_open():
            source.close()
            
        del sources[source_id]
        
        return jsonify({"success": True})


@app.route('/api/pipelines', methods=['GET'])
def get_pipelines():
    """Get a list of all pipelines."""
    pipeline_list = []
    for pipeline_id, pipeline_info in pipelines.items():
        pipeline = pipeline_info["pipeline"]
        stats = pipeline.get_stats()
        
        pipeline_list.append({
            "id": pipeline_id,
            "source_id": pipeline_info["source_id"],
            "engine_id": pipeline_info["engine_id"],
            "mode": stats["mode"],
            "frame_count": stats["frame_count"],
            "fps": stats["fps"],
            "avg_processing_time_ms": stats["avg_processing_time_ms"],
            "uptime": time.time() - pipeline_info["start_time"]
        })
    return jsonify({"pipelines": pipeline_list})


@app.route('/api/pipelines', methods=['POST'])
def create_pipeline():
    """Create a new inference pipeline."""
    data = request.json
    
    # Required fields
    if 'source_id' not in data:
        return jsonify({"error": "Source ID is required"}), 400
        
    if 'engine_id' not in data:
        return jsonify({"error": "Engine ID is required"}), 400
        
    source_id = data['source_id']
    engine_id = data['engine_id']
    pipeline_id = data.get('id', str(uuid.uuid4()))
    config = data.get('config', {})
    
    # Start the pipeline
    if start_pipeline(pipeline_id, source_id, engine_id, config):
        return jsonify({
            "id": pipeline_id,
            "source_id": source_id,
            "engine_id": engine_id,
            "status": "running"
        })
    else:
        return jsonify({"error": "Failed to start pipeline"}), 500


@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
def delete_pipeline(pipeline_id):
    """Stop and delete a pipeline."""
    if stop_pipeline(pipeline_id):
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Pipeline not found"}), 404


@app.route('/api/pipelines/<pipeline_id>/results', methods=['GET'])
def get_pipeline_results(pipeline_id):
    """Get the latest result from a pipeline."""
    if pipeline_id not in result_queues:
        return jsonify({"error": "Pipeline not found"}), 404
        
    # Get the latest result (non-blocking)
    try:
        result = result_queues[pipeline_id].get_nowait()
        result_queues[pipeline_id].put(result)  # Put it back for other requests
        
        # Prepare the response
        response = {
            "success": result.success,
            "model_name": result.model_name,
            "inference_time_ms": result.inference_time_ms,
            "preprocessing_time_ms": result.preprocessing_time_ms,
            "postprocessing_time_ms": result.postprocessing_time_ms,
            "total_time_ms": result.total_time_ms(),
            "timestamp": result.timestamp,
            "frame_id": result.frame_id
        }
        
        # Add raw outputs (convert numpy arrays to lists)
        raw_outputs = {}
        for name, array in result.raw_outputs.items():
            raw_outputs[name] = array.tolist()
            
        response["raw_outputs"] = raw_outputs
        
        # Add processed outputs if they can be converted to JSON
        if result.processed_outputs is not None:
            try:
                if isinstance(result.processed_outputs, dict):
                    processed = {}
                    for k, v in result.processed_outputs.items():
                        if isinstance(v, np.ndarray):
                            processed[k] = v.tolist()
                        else:
                            processed[k] = v
                    response["processed_outputs"] = processed
                elif isinstance(result.processed_outputs, np.ndarray):
                    response["processed_outputs"] = result.processed_outputs.tolist()
                else:
                    response["processed_outputs"] = result.processed_outputs
            except:
                response["processed_outputs"] = str(result.processed_outputs)
                
        # Add metadata
        response["metadata"] = result.metadata
        
        return jsonify(response)
        
    except queue.Empty:
        return jsonify({"error": "No results available"}), 404


@app.route('/api/pipelines/<pipeline_id>/image', methods=['GET'])
def get_pipeline_image(pipeline_id):
    """Get the latest image from a pipeline."""
    if pipeline_id not in result_queues:
        return jsonify({"error": "Pipeline not found"}), 404
        
    # Get the latest result (non-blocking)
    try:
        result = result_queues[pipeline_id].get_nowait()
        result_queues[pipeline_id].put(result)  # Put it back for other requests
        
        # Check if there's an input image
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
                    
                # Convert to BGR if it's RGB
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                # Encode as JPEG
                _, jpeg_data = cv2.imencode('.jpg', img)
                
                # Return as image
                return Response(jpeg_data.tobytes(), mimetype='image/jpeg')
                
        # If no input image or not an image-like array
        return jsonify({"error": "No image available"}), 404
        
    except queue.Empty:
        return jsonify({"error": "No results available"}), 404


@app.route('/api/infer', methods=['POST'])
def infer():
    """Perform a one-time inference on uploaded data."""
    # Check if the request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    # Get the model ID
    model_id = request.form.get('model_id')
    if not model_id:
        return jsonify({"error": "No model ID provided"}), 400
        
    # Check if the model exists
    if model_id not in engines:
        return jsonify({"error": "Model not found"}), 404
        
    try:
        # Read the image file
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform inference
        engine = engines[model_id]
        result = engine(img)
        
        # Prepare the response
        response = {
            "success": result.success,
            "model_name": result.model_name,
            "inference_time_ms": result.inference_time_ms,
            "preprocessing_time_ms": result.preprocessing_time_ms,
            "postprocessing_time_ms": result.postprocessing_time_ms,
            "total_time_ms": result.total_time_ms(),
            "timestamp": result.timestamp
        }
        
        # Add raw outputs (convert numpy arrays to lists)
        raw_outputs = {}
        for name, array in result.raw_outputs.items():
            raw_outputs[name] = array.tolist()
            
        response["raw_outputs"] = raw_outputs
        
        # Add processed outputs if they can be converted to JSON
        if result.processed_outputs is not None:
            try:
                if isinstance(result.processed_outputs, dict):
                    processed = {}
                    for k, v in result.processed_outputs.items():
                        if isinstance(v, np.ndarray):
                            processed[k] = v.tolist()
                        else:
                            processed[k] = v
                    response["processed_outputs"] = processed
                elif isinstance(result.processed_outputs, np.ndarray):
                    response["processed_outputs"] = result.processed_outputs.tolist()
                else:
                    response["processed_outputs"] = result.processed_outputs
            except:
                response["processed_outputs"] = str(result.processed_outputs)
                
        # Add metadata
        response["metadata"] = result.metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return jsonify({"error": str(e)}), 500


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Start an inference server")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to listen on"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML or JSON configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the server CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        # Load config from file if specified
        config = {}
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
            
        # Pre-load models and sources if specified in config
        if "models" in config:
            for model_config in config["models"]:
                try:
                    model_path = model_config["path"]
                    model_id = model_config.get("id", str(uuid.uuid4()))
                    engine_config = model_config.get("config", {})
                    
                    # Create the engine
                    engine = get_engine(model_path, engine_config)
                    
                    # Load the model
                    if engine.load():
                        engines[model_id] = engine
                        logger.info(f"Pre-loaded model {model_id} from {model_path}")
                    else:
                        logger.error(f"Failed to pre-load model {model_id} from {model_path}")
                        
                except Exception as e:
                    logger.error(f"Error pre-loading model: {e}")
                    
        if "sources" in config:
            for source_config in config["sources"]:
                try:
                    source_type = source_config["type"]
                    source_id = source_config.get("id", str(uuid.uuid4()))
                    source_config = source_config.get("config", {})
                    
                    # Create the source
                    source = get_source(source_type, source_id, source_config)
                    sources[source_id] = source
                    logger.info(f"Pre-created source {source_id} of type {source_type}")
                    
                except Exception as e:
                    logger.error(f"Error pre-creating source: {e}")
                    
        # Start pre-configured pipelines
        if "pipelines" in config:
            for pipeline_config in config["pipelines"]:
                try:
                    pipeline_id = pipeline_config.get("id", str(uuid.uuid4()))
                    source_id = pipeline_config["source_id"]
                    engine_id = pipeline_config["engine_id"]
                    pipe_config = pipeline_config.get("config", {})
                    
                    if start_pipeline(pipeline_id, source_id, engine_id, pipe_config):
                        logger.info(f"Pre-started pipeline {pipeline_id} with source {source_id} and engine {engine_id}")
                    else:
                        logger.error(f"Failed to pre-start pipeline {pipeline_id}")
                        
                except Exception as e:
                    logger.error(f"Error pre-starting pipeline: {e}")
                    
        # Start the server
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Clean up resources
        for pipeline_id in list(pipelines.keys()):
            stop_pipeline(pipeline_id)
            
        for source_id, source in list(sources.items()):
            if source.is_open():
                source.close()
                
        for engine_id, engine in list(engines.items()):
            if engine.is_model_loaded():
                engine.unload()


if __name__ == "__main__":
    sys.exit(main()) 