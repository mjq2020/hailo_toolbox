"""
Exception classes for the preprocessing module.

This module defines custom exceptions used throughout the preprocessing pipeline
to provide clear error messages and proper error handling.
"""

from typing import Optional, Any


class PreprocessError(Exception):
    """
    Base exception class for preprocessing errors.
    
    This is the parent class for all preprocessing-related exceptions,
    providing a common interface for error handling.
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize the preprocessing error.
        
        Args:
            message (str): Human-readable error message
            details (Optional[dict]): Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class InvalidConfigError(PreprocessError):
    """
    Exception raised when preprocessing configuration is invalid.
    
    This exception is raised when the provided configuration parameters
    are invalid, missing, or incompatible with each other.
    """
    
    def __init__(self, message: str, config_field: Optional[str] = None, 
                 provided_value: Optional[Any] = None):
        """
        Initialize the invalid configuration error.
        
        Args:
            message (str): Human-readable error message
            config_field (Optional[str]): Name of the invalid configuration field
            provided_value (Optional[Any]): The invalid value that was provided
        """
        details = {}
        if config_field:
            details["config_field"] = config_field
        if provided_value is not None:
            details["provided_value"] = provided_value
            
        super().__init__(message, details)
        self.config_field = config_field
        self.provided_value = provided_value


class ImageProcessingError(PreprocessError):
    """
    Exception raised when image processing operations fail.
    
    This exception is raised when operations like resizing, normalization,
    or data type conversion fail due to invalid input or processing errors.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 image_shape: Optional[tuple] = None):
        """
        Initialize the image processing error.
        
        Args:
            message (str): Human-readable error message
            operation (Optional[str]): Name of the failed operation
            image_shape (Optional[tuple]): Shape of the input image
        """
        details = {}
        if operation:
            details["operation"] = operation
        if image_shape:
            details["image_shape"] = image_shape
            
        super().__init__(message, details)
        self.operation = operation
        self.image_shape = image_shape


class UnsupportedFormatError(PreprocessError):
    """
    Exception raised when an unsupported image format is encountered.
    
    This exception is raised when the input image format is not supported
    by the preprocessing pipeline.
    """
    
    def __init__(self, message: str, format_type: Optional[str] = None,
                 supported_formats: Optional[list] = None):
        """
        Initialize the unsupported format error.
        
        Args:
            message (str): Human-readable error message
            format_type (Optional[str]): The unsupported format type
            supported_formats (Optional[list]): List of supported formats
        """
        details = {}
        if format_type:
            details["format_type"] = format_type
        if supported_formats:
            details["supported_formats"] = supported_formats
            
        super().__init__(message, details)
        self.format_type = format_type
        self.supported_formats = supported_formats 