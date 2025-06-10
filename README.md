# Deep Learning Toolbox

一个用于深度学习模型转换和推理的工具箱，提供了强大的功能和灵活的架构。

## 主要功能

1. **模型转换**：将PyTorch、TensorFlow等框架的模型转换为ONNX格式
2. **多种视频源支持**：支持从IP摄像头、USB摄像头、MIPI摄像头、本地视频文件等输入源进行推理
3. **自定义推理处理**：允许用户定义回调函数处理模型推理结果
4. **多路视频流**：支持单路和多路视频源的同步处理
5. **命令行工具**：提供了便捷的命令行接口

## 安装

```bash
pip install -e .
```

## 使用方法

### 模型转换

将PyTorch模型转换为ONNX格式：

```bash
dl-convert model.pth --framework pytorch --input-shape 1,3,224,224
```

将TensorFlow模型转换为ONNX格式：

```bash
dl-convert model.h5 --framework tensorflow --input-shape 1,224,224,3
```

### 模型推理

使用USB摄像头进行推理：

```bash
dl-infer model.onnx --source-type webcam --visualize
```

使用视频文件进行推理：

```bash
dl-infer model.onnx --source-type file --source-path video.mp4 --visualize
```

使用IP摄像头进行推理：

```bash
dl-infer model.onnx --source-type ip_camera --source-path rtsp://username:password@ip_address:port/stream
```

### 启动推理服务器

```bash
dl-server --config server_config.yaml
```

## 项目结构

- `hailo_tools/`: 主要包含项目的核心代码
  - `converters/`: 模型转换相关的代码
  - `inference/`: 推理引擎相关的代码
  - `sources/`: 视频源相关的代码
  - `cli/`: 命令行工具相关的代码
  - `utils/`: 工具函数和辅助模块

## 扩展

### 添加新的视频源

继承 `BaseSource` 类并实现所需的方法：

```python
from hailo_tools.sources import BaseSource, SourceType

class NewSource(BaseSource):
    def __init__(self, source_id, config=None):
        super().__init__(source_id, config)
        self.source_type = SourceType.CUSTOM
        # 初始化自定义源
        
    def open(self):
        # 实现打开源的逻辑
        pass
        
    def read(self):
        # 实现读取帧的逻辑
        pass
        
    def close(self):
        # 实现关闭源的逻辑
        pass
```

### 添加自定义推理回调

创建一个接收 `InferenceResult` 的函数：

```python
from hailo_tools.inference import InferenceResult

def custom_callback(result: InferenceResult):
    # 处理推理结果
    if result.success:
        # 处理raw_outputs或processed_outputs
        pass
    else:
        # 处理错误
        pass
```

然后，在推理时使用该回调：

```bash
dl-infer model.onnx --source-type webcam --callback mymodule:custom_callback
```

## 配置文件示例

### 服务器配置

```yaml
models:
  - id: "model1"
    path: "/path/to/model.onnx"
    config:
      input_name: "input"
      providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]

sources:
  - id: "source1"
    type: "webcam"
    config:
      device_id: 0
      resolution: [640, 480]

pipelines:
  - id: "pipeline1"
    source_id: "source1"
    engine_id: "model1"
    config:
      mode: "async"
      show_fps: true
```

## 许可证

本项目使用MIT许可证。详见LICENSE文件。 