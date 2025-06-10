# Deep Learning Image Preprocessing Module

这是一个功能强大、模块化的深度学习图像预处理系统，专为深度学习模型的输入数据预处理而设计。该模块提供了灵活的配置选项、可组合的变换操作和高性能的批处理能力。

## 主要特性

### 🚀 核心功能
- **模块化设计**: 每个预处理操作都是独立的变换类，可以单独使用或组合使用
- **配置驱动**: 通过配置类管理所有预处理参数，支持配置文件的保存和加载
- **管道组合**: 支持将多个变换操作组合成预处理管道
- **批处理支持**: 高效的批量图像处理能力
- **性能监控**: 内置时间统计功能，帮助优化预处理性能
- **错误处理**: 完善的异常处理机制，提供详细的错误信息

### 🔧 支持的变换操作
- **图像缩放**: 支持多种插值方法，可保持宽高比
- **归一化**: 支持标准归一化、最小-最大缩放等
- **数据类型转换**: 安全的数据类型转换，支持值域缩放
- **填充**: 多种填充模式（常数、反射、复制等）
- **裁剪**: 中心裁剪、指定区域裁剪
- **颜色格式转换**: BGR、RGB、灰度图之间的转换

## 快速开始

### 基本使用

```python
import numpy as np
from hailo_toolbox.process import ImagePreprocessor, PreprocessConfig

# 创建配置
config = PreprocessConfig(
    target_size=(224, 224),          # 目标尺寸
    normalize=True,                  # 启用归一化
    mean=[0.485, 0.456, 0.406],     # ImageNet均值
    std=[0.229, 0.224, 0.225],      # ImageNet标准差
    scale=1.0/255.0,                # 缩放因子
    input_format="BGR",             # 输入格式
    output_format="RGB"             # 输出格式
)

# 创建预处理器
preprocessor = ImagePreprocessor(config)

# 处理图像
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
processed_image = preprocessor(image)

print(f"原始图像: {image.shape}, {image.dtype}")
print(f"处理后: {processed_image.shape}, {processed_image.dtype}")
```

### 自定义管道

```python
from hailo_toolbox.process import (
    ResizeTransform, NormalizationTransform, 
    PreprocessPipeline
)

# 创建自定义变换
resize_transform = ResizeTransform(
    target_size=(416, 416),
    interpolation="LINEAR",
    preserve_aspect_ratio=True
)

normalize_transform = NormalizationTransform(
    mean=127.5,
    std=127.5,
    scale=1.0,
    dtype=np.float32
)

# 创建管道
pipeline = PreprocessPipeline(
    transforms=[resize_transform, normalize_transform],
    name="YOLO_Preprocessing",
    enable_timing=True
)

# 处理图像
result = pipeline(image)
pipeline.print_timing_stats()
```

## 详细配置选项

### PreprocessConfig 参数说明

```python
@dataclass
class PreprocessConfig:
    # 缩放参数
    target_size: Optional[Tuple[int, int]] = None  # 目标尺寸 (width, height)
    interpolation: str = "LINEAR"                   # 插值方法: NEAREST, LINEAR, CUBIC, AREA, LANCZOS4
    preserve_aspect_ratio: bool = False             # 是否保持宽高比
    
    # 归一化参数
    normalize: bool = True                          # 是否启用归一化
    mean: Union[float, List[float]] = 0.0          # 均值
    std: Union[float, List[float]] = 1.0           # 标准差
    scale: float = 1.0                             # 缩放因子
    
    # 数据类型参数
    target_dtype: Optional[str] = None             # 目标数据类型
    scale_values: bool = True                      # 是否缩放值域
    clip_values: bool = True                       # 是否裁剪值域
    
    # 填充参数
    padding: Optional[Union[int, Tuple]] = None    # 填充大小
    padding_mode: str = "CONSTANT"                 # 填充模式
    padding_value: Union[int, float] = 0           # 填充值
    
    # 裁剪参数
    crop_size: Optional[Tuple[int, int]] = None    # 裁剪尺寸
    crop_region: Optional[Tuple[int, int, int, int]] = None  # 裁剪区域
    center_crop: bool = True                       # 是否中心裁剪
    
    # 管道参数
    enable_timing: bool = False                    # 是否启用时间统计
    pipeline_name: str = "ImagePreprocessor"      # 管道名称
    
    # 颜色格式
    input_format: str = "BGR"                      # 输入格式: BGR, RGB, GRAY
    output_format: str = "RGB"                     # 输出格式: BGR, RGB, GRAY
```

## 使用场景示例

### 1. ImageNet 分类模型预处理

```python
# ResNet/VGG 等模型的标准预处理
config = PreprocessConfig(
    target_size=(224, 224),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    scale=1.0/255.0,
    input_format="BGR",
    output_format="RGB"
)
```

### 2. YOLO 目标检测预处理

```python
# YOLO 模型的预处理配置
config = PreprocessConfig(
    target_size=(416, 416),
    preserve_aspect_ratio=True,
    normalize=True,
    mean=0.0,
    std=255.0,
    scale=1.0,
    target_dtype="float32"
)
```

### 3. 语义分割模型预处理

```python
# 语义分割模型的预处理配置
config = PreprocessConfig(
    target_size=(512, 512),
    interpolation="CUBIC",
    normalize=True,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    scale=1.0/255.0,
    padding=(10, 10),
    padding_mode="REFLECT"
)
```

### 4. 批量处理

```python
# 批量处理多张图像
images = [image1, image2, image3, image4]
processed_images = preprocessor.process_batch(images)

# 或使用管道批量处理
processed_images = pipeline.process_batch(images)
```

## 配置管理

### 保存和加载配置

```python
# 保存配置到文件
config.save("preprocessing_config.json")

# 从文件加载配置
loaded_config = PreprocessConfig.load("preprocessing_config.json")

# 从配置文件创建预处理器
preprocessor = ImagePreprocessor.from_config_file("preprocessing_config.json")
```

### 动态更新配置

```python
# 动态更新预处理器配置
preprocessor.update_config(
    target_size=(256, 256),
    normalize=False,
    enable_timing=True
)
```

## 性能监控

### 启用时间统计

```python
# 在配置中启用时间统计
config = PreprocessConfig(enable_timing=True)
preprocessor = ImagePreprocessor(config)

# 或在管道中启用
pipeline = PreprocessPipeline(transforms, enable_timing=True)

# 处理图像后查看统计信息
preprocessor.print_timing_stats()
```

### 时间统计输出示例

```
Timing Statistics for Pipeline: ImagePreprocessor
----------------------------------------------------------------------
Transform                 Calls    Total (s)    Avg (s)    
----------------------------------------------------------------------
ResizeTransform           100      0.1234       0.0012     
NormalizationTransform    100      0.0567       0.0006     
----------------------------------------------------------------------
TOTAL                     200      0.1801       0.0009     
----------------------------------------------------------------------
```

## 错误处理

该模块提供了完善的错误处理机制：

```python
from hailo_toolbox.process import PreprocessError, InvalidConfigError

try:
    # 无效配置
    config = PreprocessConfig(target_size=(0, 224))
except InvalidConfigError as e:
    print(f"配置错误: {e}")
    print(f"错误字段: {e.config_field}")
    print(f"提供的值: {e.provided_value}")

try:
    # 处理无效输入
    result = preprocessor("not an image")
except PreprocessError as e:
    print(f"处理错误: {e}")
    print(f"错误详情: {e.details}")
```

## 扩展性

### 自定义变换

```python
from hailo_toolbox.process.transforms import BaseTransform

class CustomTransform(BaseTransform):
    def __init__(self, param1, param2, name=None):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, image):
        self.validate_image(image)
        # 实现自定义变换逻辑
        return processed_image
    
    def get_config(self):
        return {
            "param1": self.param1,
            "param2": self.param2,
            "name": self.name
        }

# 使用自定义变换
custom_transform = CustomTransform(param1="value1", param2="value2")
pipeline = PreprocessPipeline([custom_transform])
```

## 最佳实践

### 1. 性能优化
- 对于批量处理，使用 `process_batch()` 方法
- 启用时间统计来识别性能瓶颈
- 根据模型需求选择合适的插值方法

### 2. 内存管理
- 对于大图像，考虑先裁剪再缩放
- 使用适当的数据类型以节省内存

### 3. 配置管理
- 为不同模型创建专门的配置文件
- 使用有意义的管道名称便于调试

### 4. 错误处理
- 始终使用 try-catch 块处理预处理操作
- 检查输入图像的格式和尺寸

## API 参考

### 主要类

- `ImagePreprocessor`: 主要的预处理器类
- `PreprocessConfig`: 配置管理类
- `PreprocessPipeline`: 预处理管道类

### 变换类

- `ResizeTransform`: 图像缩放变换
- `NormalizationTransform`: 归一化变换
- `DataTypeTransform`: 数据类型转换变换
- `PaddingTransform`: 填充变换
- `CropTransform`: 裁剪变换

### 异常类

- `PreprocessError`: 基础预处理异常
- `InvalidConfigError`: 无效配置异常
- `ImageProcessingError`: 图像处理异常
- `UnsupportedFormatError`: 不支持格式异常

## 依赖项

- `opencv-python >= 4.5.0`
- `numpy >= 1.19.0`

## 许可证

本模块遵循项目的整体许可证。

# YOLOv8 Postprocessors

This module provides comprehensive postprocessing capabilities for YOLOv8 models, supporting detection, segmentation, and keypoint detection tasks.

## Features

### Supported Tasks
- **Object Detection**: Standard YOLOv8 detection with confidence filtering and NMS
- **Instance Segmentation**: YOLOv8-seg with mask generation from prototypes
- **Keypoint Detection**: YOLOv8-pose with pose validation and keypoint filtering

### Key Capabilities
- Configurable confidence and IoU thresholds
- Non-Maximum Suppression (NMS) filtering
- Coordinate scaling to original image dimensions
- Comprehensive error handling and validation
- Extensible architecture for future enhancements

## Quick Start

### Basic Usage

```python
from hailo_toolbox.process import create_postprocessor, PostprocessConfig

# Create configuration
config = PostprocessConfig(
    num_classes=80,
    det_conf_threshold=0.25,
    nms_iou_threshold=0.5,
    input_shape=(640, 640)
)

# Create postprocessor
postprocessor = create_postprocessor("detection", config)

# Process model outputs
result = postprocessor.postprocess(raw_outputs)
print(f"Detected {len(result)} objects")
```

### Detection Example

```python
# Detection configuration
config = PostprocessConfig(
    num_classes=80,
    det_conf_threshold=0.25,
    nms_iou_threshold=0.5,
    det_max_detections=100,
    input_shape=(640, 640),
    class_names=["person", "bicycle", "car", ...]  # COCO classes
)

postprocessor = create_postprocessor("detection", config)
result = postprocessor.postprocess(raw_outputs, original_shape=(1280, 1280))

# Access results
boxes = result.boxes          # Shape: (N, 4) - [x1, y1, x2, y2]
scores = result.scores        # Shape: (N,) - confidence scores
class_ids = result.class_ids  # Shape: (N,) - class indices
```

### Segmentation Example

```python
# Segmentation configuration
config = PostprocessConfig(
    num_classes=80,
    seg_conf_threshold=0.25,
    seg_mask_threshold=0.5,
    nms_iou_threshold=0.5,
    seg_max_instances=100,
    input_shape=(640, 640)
)

postprocessor = create_postprocessor("segmentation", config)
result = postprocessor.postprocess(raw_outputs)

# Access results
masks = result.masks          # Shape: (N, H, W) - binary masks
boxes = result.boxes          # Shape: (N, 4) - bounding boxes
scores = result.scores        # Shape: (N,) - confidence scores
class_ids = result.class_ids  # Shape: (N,) - class indices
```

### Keypoint Detection Example

```python
# Keypoint configuration
config = PostprocessConfig(
    num_keypoints=17,  # COCO pose format
    kp_conf_threshold=0.25,
    kp_visibility_threshold=0.5,
    nms_iou_threshold=0.5,
    kp_max_persons=100,
    input_shape=(640, 640),
    keypoint_names=[
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
)

postprocessor = create_postprocessor("keypoint", config)
result = postprocessor.postprocess(raw_outputs)

# Access results
keypoints = result.keypoints  # Shape: (N, K, 3) - [x, y, visibility]
boxes = result.boxes          # Shape: (N, 4) - person bounding boxes
scores = result.scores        # Shape: (N,) - person confidence scores
```

## Configuration Options

### PostprocessConfig Parameters

#### Common Parameters
- `input_shape`: Input image dimensions (height, width)
- `nms_iou_threshold`: IoU threshold for NMS (default: 0.5)

#### Detection Parameters
- `num_classes`: Number of object classes
- `det_conf_threshold`: Minimum confidence threshold (default: 0.25)
- `det_max_detections`: Maximum number of detections (default: 300)
- `class_names`: List of class names (optional)

#### Segmentation Parameters
- `seg_conf_threshold`: Minimum confidence threshold (default: 0.25)
- `seg_mask_threshold`: Mask binarization threshold (default: 0.5)
- `seg_max_instances`: Maximum number of instances (default: 100)

#### Keypoint Parameters
- `num_keypoints`: Number of keypoints per person (default: 17)
- `kp_conf_threshold`: Minimum person confidence (default: 0.25)
- `kp_visibility_threshold`: Minimum keypoint visibility (default: 0.5)
- `kp_max_persons`: Maximum number of persons (default: 100)
- `keypoint_names`: List of keypoint names (optional)

## Input Formats

### Detection Input
Expected dictionary with key `"output"` containing array of shape `(N, 4+C)`:
- Columns 0-3: `[x_center, y_center, width, height]`
- Columns 4+: Class confidence scores

### Segmentation Input
Expected dictionary with keys:
- `"output0"`: Detection array of shape `(N, 4+C+M)`
  - Columns 0-3: Bounding box coordinates
  - Columns 4 to 4+C: Class confidence scores
  - Columns 4+C to 4+C+M: Mask coefficients
- `"output1"`: Mask prototypes of shape `(M, H, W)`

### Keypoint Input
Expected dictionary with key `"output"` containing array of shape `(N, 5+K*3)`:
- Columns 0-4: `[x_center, y_center, width, height, person_confidence]`
- Columns 5+: Keypoint data `[x, y, visibility]` for each keypoint

## Output Formats

### DetectionResult
- `boxes`: Bounding boxes in `[x1, y1, x2, y2]` format
- `scores`: Confidence scores
- `class_ids`: Class indices

### SegmentationResult
- `masks`: Binary instance masks
- `boxes`: Bounding boxes
- `scores`: Confidence scores
- `class_ids`: Class indices

### KeypointResult
- `keypoints`: Keypoint coordinates and visibility `[x, y, visibility]`
- `boxes`: Person bounding boxes
- `scores`: Person confidence scores

## Advanced Features

### Coordinate Scaling
Automatically scale coordinates from model input size to original image size:

```python
result = postprocessor.postprocess(raw_outputs, original_shape=(1920, 1080))
```

### Custom Thresholds
Fine-tune detection sensitivity:

```python
# High precision (fewer false positives)
config = PostprocessConfig(det_conf_threshold=0.8, nms_iou_threshold=0.3)

# High recall (fewer missed detections)
config = PostprocessConfig(det_conf_threshold=0.1, nms_iou_threshold=0.7)
```

### Pose Validation
Keypoint postprocessor includes anatomical validation:
- Minimum visible keypoints requirement
- Pose similarity filtering
- Anatomical relationship validation

## Testing

Run the comprehensive test suite:

```bash
python -m pytest hailo_toolbox/process/test_postprocessors.py -v
```

Run example demonstrations:

```bash
python -m hailo_toolbox.process.example_usage
```

## Architecture

### Class Hierarchy
```
BasePostprocessor (Abstract)
├── YOLOv8DetPostprocessor
├── YOLOv8SegPostprocessor
└── YOLOv8KpPostprocessor
```

### Key Components
- **Configuration Management**: Centralized parameter validation
- **Result Classes**: Structured output containers
- **Factory Pattern**: Unified postprocessor creation
- **Error Handling**: Comprehensive validation and logging

## Performance Considerations

- **Vectorized Operations**: NumPy-based processing for efficiency
- **Memory Management**: Efficient array operations and memory reuse
- **Configurable Limits**: Maximum detection/instance limits to control memory usage
- **Early Filtering**: Confidence-based filtering before expensive operations

## Extension Points

The architecture supports easy extension for:
- New YOLOv8 variants (YOLOv8x, YOLOv8n, etc.)
- Custom postprocessing logic
- Additional output formats
- Domain-specific validation rules

## Dependencies

- NumPy: Numerical operations
- SciPy: Advanced mathematical functions
- Logging: Built-in Python logging

## License

This implementation follows the project's licensing terms. 