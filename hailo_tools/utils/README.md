# 日志模块使用说明

`hailo_tools.utils.logging` 模块提供了整个项目的日志记录功能，便于开发和调试。

## 基本用法

### 获取日志记录器

```python
from hailo_tools.utils import get_logger

# 获取或创建一个记录器，建议使用模块名作为记录器名称
logger = get_logger(__name__)

# 记录不同级别的日志
logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
logger.critical("这是严重错误信息")
```

### 自定义日志记录器

```python
from hailo_tools.utils import setup_logger

# 创建一个详细配置的记录器
logger = setup_logger(
    name="my_module",  # 记录器名称
    level="DEBUG",     # 日志级别
    log_file="my_custom.log",  # 日志文件名
    log_dir="custom_logs",     # 日志目录
    console=True,      # 是否输出到控制台
    max_bytes=1024*1024*5,  # 单个日志文件最大5MB
    backup_count=3     # 保留3个备份文件
)
```

## 高级功能

### 设置自定义日志格式

```python
logger = setup_logger(
    name="formatter_test",
    level="INFO",
    log_format="%(asctime)s [%(levelname)s] <%(name)s> %(message)s (%(filename)s:%(lineno)d)",
    date_format="%Y-%m-%d %H:%M:%S.%f"
)
```

### 在命令行工具中使用

```python
# cli/convert.py
from hailo_tools.utils import setup_logger

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = "DEBUG" if args.verbose else "INFO"
    
    # 配置日志
    logger = setup_logger("hailo_tools.convert", level=log_level)
    
    # 使用日志
    logger.info("开始转换模型...")
    
    # ...
```

### 子模块日志设置

```python
# 主模块设置
main_logger = setup_logger("hailo_tools.inference", level="INFO")

# 子模块继承主模块的设置
# 方式1：使用相同前缀，自动继承
onnx_logger = get_logger("hailo_tools.inference.onnx")

# 方式2：显式设置继承
pipeline_logger = setup_logger(
    "hailo_tools.inference.pipeline", 
    level="DEBUG",
    propagate=True  # 将日志传递给上级记录器
)
```

## 使用技巧

1. **合理使用日志级别**：
   - DEBUG: 详细的开发和调试信息
   - INFO: 普通的运行信息
   - WARNING: 警告但不影响运行的情况
   - ERROR: 发生错误但程序可以继续运行
   - CRITICAL: 致命错误，程序无法继续运行

2. **结构化日志**：
   ```python
   # 好的做法 - 使用结构化的消息格式
   logger.info("模型转换完成，用时: %.2f秒，输出文件: %s", 
               conversion_time, output_path)
   
   # 避免字符串拼接，对性能有影响
   # logger.info("模型转换完成，用时: " + str(conversion_time) + "秒，输出文件: " + output_path)
   ```

3. **捕获异常时记录**：
   ```python
   try:
       # 操作代码
       result = process_data()
   except Exception as e:
       logger.exception("处理数据时发生错误")  # 自动包含堆栈跟踪
       # 或
       logger.error("处理数据时发生错误: %s", str(e))
   ```

4. **使用记录器名称层次结构**：
   - 主模块：`hailo_tools`
   - 子模块：`hailo_tools.converters`
   - 具体类：`hailo_tools.converters.pytorch`

这样可以更容易地根据模块调整日志级别和处理方式。 