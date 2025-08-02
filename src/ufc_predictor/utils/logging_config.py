"""
Comprehensive Logging and Error Handling Framework
=================================================

This module provides a centralized logging configuration and structured error
handling for the UFC predictor system. It replaces inconsistent print statements
and ad-hoc error handling with professional-grade logging and exception management.

Features:
- Structured logging with multiple output formats
- File and console logging with rotation
- Contextual error information
- Performance timing integration
- Debug modes for development
- Production-ready configuration

Usage:
    from ufc_predictor.utils.logging_config import setup_logging, get_logger, UFCPredictorError
    
    # Setup logging (call once at application start)
    setup_logging(level='INFO', log_file='ufc_predictor.log')
    
    # Get logger for any module
    logger = get_logger(__name__)
    
    # Use structured logging
    logger.info("Starting prediction", extra={'fighter_a': 'Jon Jones', 'fighter_b': 'Stipe Miocic'})
"""

import logging
import logging.handlers
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import time


class UFCPredictorError(Exception):
    """Base exception for UFC Predictor specific errors"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'UFC_ERROR'
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()


class DataProcessingError(UFCPredictorError):
    """Raised when data processing fails"""
    
    def __init__(self, message: str, data_source: str = None, **kwargs):
        super().__init__(message, error_code='DATA_ERROR', **kwargs)
        self.data_source = data_source


class ModelError(UFCPredictorError):
    """Raised when model operations fail"""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, error_code='MODEL_ERROR', **kwargs)
        self.model_type = model_type


class ScrapingError(UFCPredictorError):
    """Raised when web scraping fails"""
    
    def __init__(self, message: str, url: str = None, **kwargs):
        super().__init__(message, error_code='SCRAPING_ERROR', **kwargs)
        self.url = url


class PredictionError(UFCPredictorError):
    """Raised when prediction fails"""
    
    def __init__(self, message: str, fighter_a: str = None, fighter_b: str = None, **kwargs):
        super().__init__(message, error_code='PREDICTION_ERROR', **kwargs)
        self.fighter_a = fighter_a
        self.fighter_b = fighter_b


class ContextualFormatter(logging.Formatter):
    """Custom formatter that includes contextual information in log messages"""
    
    def format(self, record):
        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.now().isoformat()
        
        # Add context information if available
        context_info = []
        for attr in ['fighter_a', 'fighter_b', 'model_type', 'data_source', 'url', 'operation']:
            if hasattr(record, attr):
                context_info.append(f"{attr}={getattr(record, attr)}")
        
        if context_info:
            record.context = f"[{', '.join(context_info)}]"
        else:
            record.context = ""
        
        return super().format(record)


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.timers[operation] = time.time()
        self.logger.debug(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, log_level: str = 'info'):
        """End timing an operation and log the duration"""
        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            del self.timers[operation]
            
            log_method = getattr(self.logger, log_level.lower())
            log_method(f"Completed operation: {operation} in {duration:.2f}s", 
                      extra={'operation': operation, 'duration': duration})
            return duration
        else:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return None
    
    @contextmanager
    def timed_operation(self, operation: str, log_level: str = 'info'):
        """Context manager for timing operations"""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, log_level)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    structured_format: bool = False
) -> logging.Logger:
    """
    Setup centralized logging configuration for the UFC predictor system
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (None for no file logging)
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        structured_format: Whether to use structured JSON format
        
    Returns:
        Configured root logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatters
    if structured_format:
        console_format = '%(timestamp)s | %(levelname)-8s | %(name)s | %(context)s %(message)s'
        file_format = '%(timestamp)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(context)s %(message)s'
    else:
        console_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(context)s %(message)s'
        file_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(context)s %(message)s'
    
    console_formatter = ContextualFormatter(console_format)
    file_formatter = ContextualFormatter(file_format)
    
    # Setup console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Setup file handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        'level': level,
        'log_file': log_file,
        'console_output': console_output,
        'structured_format': structured_format
    })
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with standardized configuration
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Ensure logger inherits root configuration
    if not logger.handlers:
        logger.propagate = True
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: Dict[str, Any] = None):
    """
    Log an exception with full context and traceback
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
    """
    context = context or {}
    
    error_info = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': traceback.format_exc(),
        **context
    }
    
    # Add UFC-specific error information if available
    if isinstance(exception, UFCPredictorError):
        error_info.update({
            'error_code': exception.error_code,
            'error_context': exception.context,
            'timestamp': exception.timestamp
        })
    
    logger.error("Exception occurred", extra=error_info)


def create_performance_logger(name: str) -> PerformanceLogger:
    """
    Create a performance logger for timing operations
    
    Args:
        name: Logger name
        
    Returns:
        Performance logger instance
    """
    logger = get_logger(name)
    return PerformanceLogger(logger)


class ErrorHandler:
    """Centralized error handling with context management"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @contextmanager
    def handle_errors(self, operation: str, context: Dict[str, Any] = None):
        """
        Context manager for handling errors in operations
        
        Args:
            operation: Description of the operation being performed
            context: Additional context information
        """
        context = context or {}
        context['operation'] = operation
        
        try:
            self.logger.debug(f"Starting operation: {operation}", extra=context)
            yield
            self.logger.debug(f"Completed operation: {operation}", extra=context)
            
        except UFCPredictorError as e:
            # Log UFC-specific errors with full context
            log_exception(self.logger, e, context)
            raise
            
        except Exception as e:
            # Convert generic exceptions to UFC-specific ones
            ufc_error = UFCPredictorError(
                f"Operation failed: {operation} - {str(e)}",
                context=context
            )
            log_exception(self.logger, ufc_error, context)
            raise ufc_error from e
    
    def safe_execute(self, operation: str, func, *args, **kwargs):
        """
        Safely execute a function with error handling
        
        Args:
            operation: Description of the operation
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result or None if error occurred
        """
        try:
            with self.handle_errors(operation):
                return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Safe execution failed for {operation}: {e}")
            return None


# Global logging utilities
def configure_for_development():
    """Configure logging for development environment"""
    setup_logging(
        level='DEBUG',
        log_file='logs/ufc_predictor_dev.log',
        console_output=True,
        structured_format=False
    )


def configure_for_production():
    """Configure logging for production environment"""
    setup_logging(
        level='INFO',
        log_file='logs/ufc_predictor_prod.log',
        console_output=False,
        structured_format=True
    )


def configure_for_testing():
    """Configure logging for testing environment"""
    setup_logging(
        level='WARNING',
        log_file=None,  # No file logging during tests
        console_output=False,  # Suppress console output during tests
        structured_format=False
    )


# Initialize default logging if not already configured
if not logging.getLogger().handlers:
    setup_logging()


# Example usage
if __name__ == "__main__":
    print("🚀 UFC Predictor Logging Framework Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level='DEBUG', log_file='demo.log')
    
    # Get loggers
    logger = get_logger(__name__)
    perf_logger = create_performance_logger(__name__)
    error_handler = ErrorHandler(logger)
    
    # Demo structured logging
    logger.info("Starting UFC prediction", extra={
        'fighter_a': 'Jon Jones',
        'fighter_b': 'Stipe Miocic',
        'operation': 'prediction'
    })
    
    # Demo performance logging
    with perf_logger.timed_operation('demo_operation'):
        time.sleep(0.1)  # Simulate work
    
    # Demo error handling
    try:
        with error_handler.handle_errors('demo_error', {'test': True}):
            raise ValueError("Demo error")
    except UFCPredictorError as e:
        print(f"Caught UFC error: {e.error_code}")
    
    logger.info("Demo completed successfully")
    print("\n✅ Logging framework demo completed")
    print("Check 'demo.log' for file output")