"""
Unified Data Interface System for Deep Learning Inference Pipeline

This module defines the core data structures and interfaces that enable seamless
data flow between different stages of the inference pipeline: source -> preprocessing 
-> inference -> postprocessing -> visualization.

The design follows the principle of separation of concerns while maintaining
type safety and extensibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple, Generic, TypeVar
from enum import Enum
import numpy as np
from pathlib import Path
import time


# Type variables for generic interfaces
T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class DataType(Enum):
    """Enumeration of supported data types in the pipeline."""
    IMAGE = "image"
    VIDEO = "video"
    BATCH = "batch"
    TENSOR = "tensor"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    KEYPOINT = "keypoint"
    MULTIMODAL = "multimodal"


class ProcessingStage(Enum):
    """Enumeration of processing stages in the pipeline."""
    SOURCE = "source"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    VISUALIZATION = "visualization"
    OUTPUT = "output"


@dataclass
class Metadata:
    """
    Metadata container for tracking data provenance and processing information.
    """
    # Source information
    source_path: Optional[str] = None
    source_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    # Processing information
    processing_stage: ProcessingStage = ProcessingStage.SOURCE
    processing_history: List[str] = field(default_factory=list)
    
    # Data properties
    data_type: DataType = DataType.IMAGE
    original_shape: Optional[Tuple[int, ...]] = None
    current_shape: Optional[Tuple[int, ...]] = None
    
    # Model information
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    
    # Performance metrics
    processing_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Optional[float] = None
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def add_processing_step(self, step_name: str, duration: float = None):
        """Add a processing step to the history."""
        self.processing_history.append(f"{step_name}@{time.time()}")
        if duration is not None:
            self.processing_times[step_name] = duration
    
    def update_stage(self, stage: ProcessingStage):
        """Update the current processing stage."""
        self.processing_stage = stage
    
    def copy(self) -> 'Metadata':
        """Create a deep copy of metadata."""
        return Metadata(
            source_path=self.source_path,
            source_type=self.source_type,
            timestamp=self.timestamp,
            processing_stage=self.processing_stage,
            processing_history=self.processing_history.copy(),
            data_type=self.data_type,
            original_shape=self.original_shape,
            current_shape=self.current_shape,
            model_name=self.model_name,
            model_version=self.model_version,
            processing_times=self.processing_times.copy(),
            memory_usage=self.memory_usage,
            custom_attributes=self.custom_attributes.copy()
        )


@dataclass
class DataContainer(Generic[T]):
    """
    Universal data container that wraps data with metadata and provides
    a consistent interface across all pipeline stages.
    """
    data: T
    metadata: Metadata = field(default_factory=Metadata)
    
    # Additional data fields for different stages
    raw_data: Optional[Any] = None  # Original raw data
    preprocessed_data: Optional[np.ndarray] = None  # Preprocessed tensor
    inference_outputs: Optional[Dict[str, np.ndarray]] = None  # Raw model outputs
    postprocessed_results: Optional[Any] = None  # Structured results
    visualization_data: Optional[Any] = None  # Visualization outputs
    
    def __post_init__(self):
        """Initialize container after creation."""
        if hasattr(self.data, 'shape'):
            self.metadata.current_shape = self.data.shape
            if self.metadata.original_shape is None:
                self.metadata.original_shape = self.data.shape
    
    def update_data(self, new_data: Any, stage: ProcessingStage):
        """Update data and metadata for a new processing stage."""
        self.data = new_data
        self.metadata.update_stage(stage)
        if hasattr(new_data, 'shape'):
            self.metadata.current_shape = new_data.shape
    
    def add_inference_output(self, outputs: Dict[str, np.ndarray]):
        """Add inference outputs to the container."""
        self.inference_outputs = outputs
        self.metadata.update_stage(ProcessingStage.INFERENCE)
    
    def add_postprocessed_results(self, results: Any):
        """Add postprocessed results to the container."""
        self.postprocessed_results = results
        self.metadata.update_stage(ProcessingStage.POSTPROCESSING)
    
    def add_visualization_data(self, vis_data: Any):
        """Add visualization data to the container."""
        self.visualization_data = vis_data
        self.metadata.update_stage(ProcessingStage.VISUALIZATION)
    
    def get_current_data(self) -> Any:
        """Get the most recent data based on processing stage."""
        stage = self.metadata.processing_stage
        
        if stage == ProcessingStage.VISUALIZATION and self.visualization_data is not None:
            return self.visualization_data
        elif stage == ProcessingStage.POSTPROCESSING and self.postprocessed_results is not None:
            return self.postprocessed_results
        elif stage == ProcessingStage.INFERENCE and self.inference_outputs is not None:
            return self.inference_outputs
        elif stage == ProcessingStage.PREPROCESSING and self.preprocessed_data is not None:
            return self.preprocessed_data
        else:
            return self.data
    
    def clone(self) -> 'DataContainer[T]':
        """Create a deep copy of the container."""
        return DataContainer(
            data=self.data,
            metadata=self.metadata.copy(),
            raw_data=self.raw_data,
            preprocessed_data=self.preprocessed_data,
            inference_outputs=self.inference_outputs,
            postprocessed_results=self.postprocessed_results,
            visualization_data=self.visualization_data
        )


class DataInterface(ABC, Generic[InputType, OutputType]):
    """
    Abstract base class for all pipeline components.
    
    This interface ensures consistent data handling across all stages
    of the inference pipeline.
    """
    
    @abstractmethod
    def process(self, input_data: DataContainer[InputType]) -> DataContainer[OutputType]:
        """
        Process input data and return output data.
        
        Args:
            input_data: Input data container
            
        Returns:
            Output data container with updated metadata
        """
        pass
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        """Get the processing stage this component represents."""
        pass
    
    def __call__(self, input_data: DataContainer[InputType]) -> DataContainer[OutputType]:
        """Callable interface for processing."""
        start_time = time.time()
        
        # Process the data
        result = self.process(input_data)
        
        # Update metadata with processing information
        duration = time.time() - start_time
        stage_name = self.get_stage().value
        result.metadata.add_processing_step(stage_name, duration)
        
        return result


class SourceInterface(DataInterface[None, Any]):
    """Interface for data sources (cameras, files, streams, etc.)."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.SOURCE
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Get source properties (resolution, fps, format, etc.)."""
        pass


class PreprocessorInterface(DataInterface[Any, np.ndarray]):
    """Interface for data preprocessing components."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.PREPROCESSING
    
    @abstractmethod
    def get_input_spec(self) -> Dict[str, Any]:
        """Get input specification requirements."""
        pass
    
    @abstractmethod
    def get_output_spec(self) -> Dict[str, Any]:
        """Get output specification."""
        pass


class InferenceInterface(DataInterface[np.ndarray, Dict[str, np.ndarray]]):
    """Interface for model inference components."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.INFERENCE
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (name, version, input/output specs)."""
        pass
    
    @abstractmethod
    def warm_up(self, input_shape: Tuple[int, ...]) -> None:
        """Warm up the inference engine with dummy data."""
        pass


class PostprocessorInterface(DataInterface[Dict[str, np.ndarray], Any]):
    """Interface for postprocessing components."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.POSTPROCESSING
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types."""
        pass


class VisualizationInterface(DataInterface[Any, Any]):
    """Interface for visualization components."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.VISUALIZATION
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        pass


class OutputInterface(DataInterface[Any, Any]):
    """Interface for output components (file writers, streamers, etc.)."""
    
    @abstractmethod
    def get_stage(self) -> ProcessingStage:
        return ProcessingStage.OUTPUT
    
    @abstractmethod
    def finalize(self) -> None:
        """Finalize output (close files, cleanup resources, etc.)."""
        pass


@dataclass
class PipelineConfig:
    """Configuration for the entire inference pipeline."""
    
    # Source configuration
    source_config: Dict[str, Any] = field(default_factory=dict)
    
    # Preprocessing configuration
    preprocess_config: Dict[str, Any] = field(default_factory=dict)
    
    # Inference configuration
    inference_config: Dict[str, Any] = field(default_factory=dict)
    
    # Postprocessing configuration
    postprocess_config: Dict[str, Any] = field(default_factory=dict)
    
    # Visualization configuration
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    
    # Output configuration
    output_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline-wide settings
    batch_size: int = 1
    enable_profiling: bool = False
    enable_caching: bool = False
    max_queue_size: int = 10
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    
    def validate(self) -> bool:
        """Validate the pipeline configuration."""
        # Add validation logic here
        return True


class Pipeline:
    """
    Main pipeline orchestrator that coordinates data flow between components.
    
    This class manages the entire inference pipeline and ensures proper
    data flow and error handling.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.components: Dict[ProcessingStage, DataInterface] = {}
        self.is_initialized = False
        
        # Performance tracking
        self.total_processed = 0
        self.total_errors = 0
        self.processing_times: Dict[str, List[float]] = {}
    
    def add_component(self, component: DataInterface) -> 'Pipeline':
        """
        Add a component to the pipeline.
        
        Args:
            component: Pipeline component
            
        Returns:
            Self for method chaining
        """
        stage = component.get_stage()
        self.components[stage] = component
        return self
    
    def initialize(self) -> None:
        """Initialize all pipeline components."""
        if not self.config.validate():
            raise ValueError("Invalid pipeline configuration")
        
        # Initialize components in order
        required_stages = [
            ProcessingStage.SOURCE,
            ProcessingStage.PREPROCESSING,
            ProcessingStage.INFERENCE,
            ProcessingStage.POSTPROCESSING
        ]
        
        for stage in required_stages:
            if stage not in self.components:
                raise ValueError(f"Missing required component for stage: {stage}")
        
        # Warm up inference engine if available
        if ProcessingStage.INFERENCE in self.components:
            inference_component = self.components[ProcessingStage.INFERENCE]
            if hasattr(inference_component, 'warm_up'):
                # Get input shape from preprocessing component
                preprocess_component = self.components[ProcessingStage.PREPROCESSING]
                output_spec = preprocess_component.get_output_spec()
                input_shape = output_spec.get('shape', (1, 3, 640, 640))
                inference_component.warm_up(input_shape)
        
        self.is_initialized = True
    
    def process_single(self, input_data: Any = None) -> DataContainer:
        """
        Process a single data item through the pipeline.
        
        Args:
            input_data: Input data (None for source-driven pipelines)
            
        Returns:
            Final processed data container
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Start with source
            source = self.components[ProcessingStage.SOURCE]
            if input_data is None:
                container = source.process(DataContainer(data=None))
            else:
                container = DataContainer(data=input_data)
                container.metadata.update_stage(ProcessingStage.SOURCE)
            
            # Process through each stage
            stages = [
                ProcessingStage.PREPROCESSING,
                ProcessingStage.INFERENCE,
                ProcessingStage.POSTPROCESSING
            ]
            
            for stage in stages:
                if stage in self.components:
                    component = self.components[stage]
                    container = component(container)
            
            # Optional visualization
            if ProcessingStage.VISUALIZATION in self.components:
                vis_component = self.components[ProcessingStage.VISUALIZATION]
                container = vis_component(container)
            
            # Optional output
            if ProcessingStage.OUTPUT in self.components:
                output_component = self.components[ProcessingStage.OUTPUT]
                container = output_component(container)
            
            self.total_processed += 1
            return container
            
        except Exception as e:
            self.total_errors += 1
            if not self.config.continue_on_error:
                raise
            else:
                # Log error and return empty container
                print(f"Pipeline error: {e}")
                return DataContainer(data=None)
    
    def process_batch(self, input_batch: List[Any]) -> List[DataContainer]:
        """
        Process a batch of data items.
        
        Args:
            input_batch: List of input data items
            
        Returns:
            List of processed data containers
        """
        results = []
        for item in input_batch:
            result = self.process_single(item)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return {
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(1, self.total_processed),
            'processing_times': self.processing_times,
            'components': list(self.components.keys())
        }
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        for component in self.components.values():
            if hasattr(component, 'cleanup'):
                component.cleanup()
        
        # Finalize output components
        if ProcessingStage.OUTPUT in self.components:
            output_component = self.components[ProcessingStage.OUTPUT]
            if hasattr(output_component, 'finalize'):
                output_component.finalize()


# Utility functions for creating common data containers
def create_image_container(image: np.ndarray, 
                          source_path: str = None,
                          metadata: Metadata = None) -> DataContainer[np.ndarray]:
    """Create a data container for image data."""
    if metadata is None:
        metadata = Metadata(
            source_path=source_path,
            data_type=DataType.IMAGE,
            original_shape=image.shape,
            current_shape=image.shape
        )
    
    return DataContainer(data=image, metadata=metadata)


def create_batch_container(batch: List[np.ndarray],
                          metadata: Metadata = None) -> DataContainer[List[np.ndarray]]:
    """Create a data container for batch data."""
    if metadata is None:
        metadata = Metadata(
            data_type=DataType.BATCH,
            original_shape=(len(batch),) + batch[0].shape if batch else (0,),
            current_shape=(len(batch),) + batch[0].shape if batch else (0,)
        )
    
    return DataContainer(data=batch, metadata=metadata)


def create_tensor_container(tensor: np.ndarray,
                           metadata: Metadata = None) -> DataContainer[np.ndarray]:
    """Create a data container for tensor data."""
    if metadata is None:
        metadata = Metadata(
            data_type=DataType.TENSOR,
            original_shape=tensor.shape,
            current_shape=tensor.shape
        )
    
    return DataContainer(data=tensor, metadata=metadata)
