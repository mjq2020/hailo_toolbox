# Hailo Toolbox Inference Tutorial

This comprehensive tutorial will guide you through running inference with the Hailo Toolbox framework. The toolbox supports both Hailo (.hef) and ONNX models with various input sources and customizable processing pipelines.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Basic Usage](#basic-usage)
- [Command Line Arguments](#command-line-arguments)
- [Supported Models](#supported-models)
- [Input Sources](#input-sources)
- [Callback Functions](#callback-functions)
- [Advanced Usage](#advanced-usage)
- [Output Management](#output-management)
- [Performance Optimization](#performance-optimization)

## Prerequisites

### Installation

Ensure you have installed the Hailo Toolbox:

```bash
pip install -e .
```

### Required Dependencies

- Python 3.7 or higher
- OpenCV (cv2)
- NumPy
- Hailo SDK (for .hef models)
- ONNX Runtime (for .onnx models)

### Hardware Requirements

- For Hailo models (.hef): Hailo-8 or compatible hardware
- For ONNX models: CPU or GPU (depending on configured providers)

## Project Structure

The inference system follows a modular architecture:

```
hailo_toolbox/
├── inference/              # Core inference engines
│   ├── base.py            # Base inference engine interface
│   ├── hailo_engine.py    # Hailo-specific inference engine
│   ├── onnx_engine.py     # ONNX inference engine
│   └── pipeline.py        # Inference pipeline orchestrator
├── process/               # Processing components
│   ├── preprocessing/     # Input preprocessing
│   ├── postprocessing/    # Output postprocessing
│   └── visualization/     # Result visualization
├── sources/               # Input source handlers
├── cli/                   # Command-line interface
└── utils/                 # Utility functions and configurations
```

## Basic Usage

### Simple Video Inference

Run inference on a video file with a Hailo model:

```bash
python examples/hailo_detect.py infer models/yolov8n.hef -c yolov8det --source sources/test.mp4
```

### Break Down of the Command

- `python examples/hailo_detect.py`: Execute the inference script
- `infer`: Subcommand to run inference (vs. convert)
- `models/yolov8n.hef`: Path to the model file
- `-c yolov8det`: Callback function name for processing
- `--source sources/test.mp4`: Input video file

## Command Line Arguments

### Core Arguments

#### Required Arguments

- `model`: Path to the model file (.hef or .onnx format)

#### Optional Arguments

- `--callback` / `-c`: Callback function name for custom processing
- `--source` / `-s`: Input source (video file, image, or camera)
- `--task-type` / `-tt`: Task type (detection, segmentation, keypoint)
- `--save` / `-sv`: Save output video (flag)
- `--save-path` / `-sp`: Path to save output video
- `--show` / `-sh`: Display output in real-time (flag)

### Complete Command Structure

```bash
python examples/hailo_detect.py infer <model_path> [OPTIONS]
```

### Examples of Different Configurations

#### 1. Object Detection with YOLOv8

```bash
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4 \
  --task-type detection \
  --save \
  --save-path output/detection_result.mp4
```

#### 2. Segmentation with YOLOv8

```bash
python examples/hailo_detect.py infer models/yolov8n_seg.hef \
  -c yolov8seg \
  --source sources/test.mp4 \
  --task-type segmentation \
  --show
```

#### 3. Pose Estimation

```bash
python examples/hailo_detect.py infer models/yolov8s_pose.hef \
  -c yolov8pose \
  --source sources/test.mp4 \
  --task-type keypoint \
  --save-path output/pose_result.mp4
```

#### 4. Real-time Webcam Inference

```bash
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source 0 \
  --show
```

## Supported Models

### Hailo Models (.hef)

The project includes several pre-trained Hailo models:

- **YOLOv8 Detection**: `yolov8n.hef`, `yolov8s.hef`, `yolov11s.hef`
- **YOLOv8 Segmentation**: `yolov8n_seg.hef`, `yolov5n_seg.hef`
- **YOLOv8 Pose**: `yolov8s_pose.hef`
- **Face Detection**: `scrfd_2.5g.hef`
- **Pose Estimation**: `centerpose_regnetx_1.6gf_fpn.hef`

### ONNX Models

The framework also supports ONNX models with automatic provider selection:

```bash
python examples/hailo_detect.py infer model.onnx \
  -c custom_callback \
  --source input.mp4
```

## Input Sources

### Supported Source Types

1. **Video Files**: MP4, AVI, MOV, etc.
   ```bash
   --source path/to/video.mp4
   ```

2. **Image Files**: JPG, PNG, BMP, etc.
   ```bash
   --source path/to/image.jpg
   ```

3. **Webcam**: USB camera (device ID)
   ```bash
   --source 0  # Default camera
   --source 1  # Secondary camera
   ```

4. **IP Camera**: RTSP streams
   ```bash
   --source rtsp://username:password@ip:port/stream
   ```

5. **MIPI Camera**: For embedded systems
   ```bash
   --source mipi://0
   ```

### Source Configuration Examples

#### High-Resolution Video Processing

```bash
python examples/hailo_detect.py infer models/yolov8s.hef \
  -c yolov8det \
  --source high_res_video.mp4 \
  --save-path output/high_res_output.mp4
```

#### Batch Image Processing

```bash
for img in images/*.jpg; do
  python examples/hailo_detect.py infer models/yolov8n.hef \
    -c yolov8det \
    --source "$img" \
    --save-path "output/$(basename "$img")"
done
```

## Callback Functions

### Understanding Callbacks

Callbacks define how the inference results are processed and visualized. They handle:

- **Preprocessing**: Input data preparation
- **Postprocessing**: Raw model output processing
- **Visualization**: Result rendering and display

### Built-in Callbacks

- `yolov8det`: YOLOv8 object detection
- `yolov8seg`: YOLOv8 segmentation
- `yolov8pose`: YOLOv8 pose estimation
- `base`: Basic processing (default)

### Custom Callback Development

Create custom callbacks by implementing the callback interface:

```python
from hailo_toolbox.inference import InferenceResult

def custom_callback(result: InferenceResult):
    """
    Custom callback for processing inference results.
    
    Args:
        result: InferenceResult containing model outputs
    """
    if result.success:
        # Process raw outputs
        raw_outputs = result.raw_outputs
        
        # Custom postprocessing logic
        processed_data = custom_postprocess(raw_outputs)
        
        # Custom visualization
        visualized_frame = custom_visualize(processed_data)
        
        # Save or display results
        save_result(visualized_frame)
    else:
        print(f"Inference failed: {result.metadata.get('error', 'Unknown error')}")
```

## Advanced Usage

### Configuration-based Inference

Create a configuration file for complex setups:

```python
# inference_config.py
from hailo_toolbox.utils.config import InferConfig

config = InferConfig({
    'model': 'models/yolov8n.hef',
    'source': 'sources/test.mp4',
    'callback': 'yolov8det',
    'preprocess': {
        'resize': (640, 640),
        'normalize': True,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'postprocess': {
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'max_detections': 100
    },
    'visualization': {
        'show_confidence': True,
        'show_labels': True,
        'thickness': 2
    }
})
```

### Multi-Model Pipeline

Process with multiple models sequentially:

```python
# Run detection first
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source input.mp4 \
  --save-path temp_detection.mp4

# Then run pose estimation on detected regions
python examples/hailo_detect.py infer models/yolov8s_pose.hef \
  -c yolov8pose \
  --source temp_detection.mp4 \
  --save-path final_output.mp4
```

### Performance Monitoring

Enable detailed performance logging:

```bash
HAILO_TOOLBOX_LOG_LEVEL=DEBUG python examples/hailo_detect.py infer \
  models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4
```

## Output Management

### Automatic Output Generation

By default, the inference generates frame-by-frame outputs:

```
output/
├── output_frame_0000.jpg
├── output_frame_0001.jpg
├── output_frame_0002.jpg
└── ...
```

### Custom Output Paths

Specify custom output locations:

```bash
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4 \
  --save \
  --save-path custom_output/results.mp4
```

### Output Formats

Supported output formats:
- **Images**: JPG, PNG, BMP
- **Videos**: MP4, AVI, MOV
- **Data**: JSON, CSV (for analysis)

### Batch Processing Script

Create a batch processing script:

```bash
#!/bin/bash
# batch_inference.sh

MODEL="models/yolov8n.hef"
CALLBACK="yolov8det"
INPUT_DIR="input_videos"
OUTPUT_DIR="output_results"

mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Processing $filename..."
    
    python examples/hailo_detect.py infer "$MODEL" \
      -c "$CALLBACK" \
      --source "$video" \
      --save \
      --save-path "$OUTPUT_DIR/${filename}_result.mp4"
done

echo "Batch processing completed!"
```

## Performance Optimization

### Hardware Acceleration

#### Hailo Hardware

For optimal Hailo performance:

```bash
# Ensure Hailo driver is loaded
sudo modprobe hailo_pci

# Check Hailo device status
hailortcli fw-control identify

# Set high performance mode
export HAILO_SCHEDULING_ALGORITHM=ROUND_ROBIN
```

#### ONNX Runtime Optimization

Configure ONNX providers for your hardware:

```python
# CPU optimization
providers = ['CPUExecutionProvider']

# GPU acceleration (if available)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Intel OpenVINO (if available)
providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
```

### Memory Management

For large video processing:

```bash
# Limit memory usage
export HAILO_TOOLBOX_MAX_MEMORY=4GB

# Use memory mapping for large files
export HAILO_TOOLBOX_USE_MMAP=true
```

### Parallel Processing

Process multiple streams simultaneously:

```bash
# Process multiple videos in parallel
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source sources/stream1.mp4 &

python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source sources/stream2.mp4 &

wait  # Wait for all processes to complete
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors

```bash
# Check model file existence and permissions
ls -la models/yolov8n.hef

# Verify model format
file models/yolov8n.hef
```

#### 2. Source Access Problems

```bash
# Test video file accessibility
ffprobe sources/test.mp4

# Check camera availability
v4l2-ctl --list-devices
```

#### 3. Memory Issues

```bash
# Monitor memory usage
watch -n 1 'free -h && nvidia-smi'

# Reduce batch size or input resolution
python examples/hailo_detect.py infer models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4 \
  --input-size 320
```

#### 4. Performance Issues

```bash
# Profile the inference
python -m cProfile examples/hailo_detect.py infer \
  models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4
```

### Debug Mode

Enable comprehensive debugging:

```bash
export HAILO_TOOLBOX_DEBUG=1
export HAILO_TOOLBOX_VERBOSE=1

python examples/hailo_detect.py infer \
  models/yolov8n.hef \
  -c yolov8det \
  --source sources/test.mp4
```

## Best Practices

### 1. Model Selection

- Use smaller models (yolov8n) for real-time applications
- Use larger models (yolov8s, yolov11s) for higher accuracy
- Choose task-specific models (detection, segmentation, pose)

### 2. Input Optimization

- Resize input to match model input size
- Use appropriate color space (RGB/BGR)
- Normalize input data consistently

### 3. Output Management

- Regularly clean output directories
- Use appropriate compression for saved videos
- Consider storage limitations for long-running processes

### 4. Resource Management

- Monitor system resources during inference
- Use appropriate batch sizes
- Clean up resources properly after processing

## Conclusion

This tutorial covers the comprehensive usage of the Hailo Toolbox inference system. The framework provides flexible, high-performance inference capabilities for various deep learning tasks. By following these guidelines and examples, you can effectively leverage the toolbox for your specific use cases.

For advanced customization and development, refer to the source code in the `hailo_toolbox/` directory and extend the base classes to implement custom functionality. 