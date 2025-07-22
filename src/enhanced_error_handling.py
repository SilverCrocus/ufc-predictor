"""
Enhanced Error Handling for UFC Predictor System
===============================================

This module provides comprehensive error handling improvements for the existing
codebase, replacing print statements with proper logging and adding structured
exception handling throughout the system.

Key improvements:
- Replace all print statements with structured logging
- Add contextual error information
- Implement retry mechanisms for transient failures
- Provide detailed error reporting for troubleshooting
- Add error recovery strategies

Usage:
    # Replace existing imports
    from src.enhanced_error_handling import (
        enhanced_prediction_wrapper,
        enhanced_scraping_wrapper,
        enhanced_model_training_wrapper
    )
    
    # Use as decorators or context managers
    @enhanced_prediction_wrapper
    def predict_fight(fighter_a, fighter_b):
        # Your prediction code
"""

import functools
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
import pandas as pd
import requests
from pathlib import Path

from .logging_config import (
    get_logger, 
    log_exception, 
    UFCPredictorError,
    DataProcessingError,
    ModelError,
    ScrapingError,
    PredictionError,
    ErrorHandler
)

logger = get_logger(__name__)


class RetryConfig:
    """Configuration for retry operations"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions


def retry_on_failure(config: RetryConfig):
    """Decorator to retry operations on failure with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.delay
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Operation succeeded on attempt {attempt + 1}", extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'total_attempts': config.max_attempts
                        })
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    if attempt < config.max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s", extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'error': str(e),
                            'delay': delay
                        })
                        time.sleep(delay)
                        delay *= config.backoff_factor
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed", extra={
                            'function': func.__name__,
                            'final_error': str(e)
                        })
            
            # Re-raise the last exception if all attempts failed
            raise last_exception
            
        return wrapper
    return decorator


def enhanced_prediction_wrapper(func: Callable) -> Callable:
    """Enhanced wrapper for prediction functions with proper error handling"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = ErrorHandler(logger)
        
        # Extract fighter names for context
        fighter_a = args[0] if len(args) > 0 else kwargs.get('fighter_a', 'Unknown')
        fighter_b = args[1] if len(args) > 1 else kwargs.get('fighter_b', 'Unknown')
        
        context = {
            'fighter_a': fighter_a,
            'fighter_b': fighter_b,
            'function': func.__name__
        }
        
        try:
            with error_handler.handle_errors('fight_prediction', context):
                logger.info(f"Starting prediction for {fighter_a} vs {fighter_b}")
                
                # Validate inputs
                if not fighter_a or not fighter_b:
                    raise PredictionError(
                        "Fighter names cannot be empty",
                        fighter_a=fighter_a,
                        fighter_b=fighter_b
                    )
                
                if fighter_a == fighter_b:
                    raise PredictionError(
                        "Cannot predict fight between same fighter",
                        fighter_a=fighter_a,
                        fighter_b=fighter_b
                    )
                
                result = func(*args, **kwargs)
                
                # Validate result
                if isinstance(result, dict) and 'error' in result:
                    raise PredictionError(
                        f"Prediction returned error: {result['error']}",
                        fighter_a=fighter_a,
                        fighter_b=fighter_b
                    )
                
                logger.info(f"Prediction completed successfully for {fighter_a} vs {fighter_b}")
                return result
                
        except Exception as e:
            if not isinstance(e, UFCPredictorError):
                e = PredictionError(
                    f"Prediction failed: {str(e)}",
                    fighter_a=fighter_a,
                    fighter_b=fighter_b
                )
            raise e
    
    return wrapper


def enhanced_scraping_wrapper(func: Callable) -> Callable:
    """Enhanced wrapper for web scraping functions with retry and error handling"""
    
    # Configure retry for common network issues
    retry_config = RetryConfig(
        max_attempts=3,
        delay=2.0,
        backoff_factor=2.0,
        exceptions=(requests.RequestException, TimeoutError, ConnectionError)
    )
    
    @retry_on_failure(retry_config)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = ErrorHandler(logger)
        
        # Extract URL for context if available
        url = None
        if args and hasattr(args[0], 'url'):
            url = args[0].url
        elif 'url' in kwargs:
            url = kwargs['url']
        
        context = {
            'function': func.__name__,
            'url': url
        }
        
        try:
            with error_handler.handle_errors('web_scraping', context):
                logger.info(f"Starting web scraping operation: {func.__name__}")
                
                result = func(*args, **kwargs)
                
                # Validate scraping result
                if result is None:
                    raise ScrapingError(
                        "Scraping returned no data",
                        url=url
                    )
                
                # Check for empty results
                if isinstance(result, (list, dict)) and len(result) == 0:
                    logger.warning("Scraping returned empty result", extra=context)
                
                logger.info(f"Web scraping completed successfully: {func.__name__}")
                return result
                
        except Exception as e:
            if not isinstance(e, UFCPredictorError):
                e = ScrapingError(
                    f"Web scraping failed: {str(e)}",
                    url=url
                )
            raise e
    
    return wrapper


def enhanced_model_training_wrapper(func: Callable) -> Callable:
    """Enhanced wrapper for model training functions with monitoring"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = ErrorHandler(logger)
        
        # Extract model type for context
        model_type = kwargs.get('model_type', 'unknown')
        if hasattr(args[0], '__class__'):
            model_type = args[0].__class__.__name__
        
        context = {
            'function': func.__name__,
            'model_type': model_type
        }
        
        try:
            with error_handler.handle_errors('model_training', context):
                logger.info(f"Starting model training: {model_type}")
                
                # Validate training data if provided
                if len(args) >= 2:
                    X, y = args[0], args[1]
                    if hasattr(X, 'shape') and hasattr(y, 'shape'):
                        if len(X) == 0 or len(y) == 0:
                            raise ModelError(
                                "Training data is empty",
                                model_type=model_type
                            )
                        
                        if len(X) != len(y):
                            raise ModelError(
                                f"Feature matrix ({len(X)}) and target vector ({len(y)}) have different lengths",
                                model_type=model_type
                            )
                        
                        logger.info(f"Training data validated: {len(X)} samples, {X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 'unknown'} features")
                
                result = func(*args, **kwargs)
                
                logger.info(f"Model training completed successfully: {model_type}")
                return result
                
        except Exception as e:
            if not isinstance(e, UFCPredictorError):
                e = ModelError(
                    f"Model training failed: {str(e)}",
                    model_type=model_type
                )
            raise e
    
    return wrapper


def enhanced_data_processing_wrapper(func: Callable) -> Callable:
    """Enhanced wrapper for data processing functions"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = ErrorHandler(logger)
        
        # Extract data source information
        data_source = 'unknown'
        if args and hasattr(args[0], 'name'):
            data_source = args[0].name
        elif 'data_source' in kwargs:
            data_source = kwargs['data_source']
        
        context = {
            'function': func.__name__,
            'data_source': data_source
        }
        
        try:
            with error_handler.handle_errors('data_processing', context):
                logger.info(f"Starting data processing: {func.__name__}")
                
                result = func(*args, **kwargs)
                
                # Validate processing result
                if result is None:
                    raise DataProcessingError(
                        "Data processing returned None",
                        data_source=data_source
                    )
                
                # Log data processing statistics
                if isinstance(result, pd.DataFrame):
                    logger.info(f"Data processing completed: {len(result)} rows, {len(result.columns)} columns", 
                               extra={**context, 'rows': len(result), 'columns': len(result.columns)})
                
                return result
                
        except Exception as e:
            if not isinstance(e, UFCPredictorError):
                e = DataProcessingError(
                    f"Data processing failed: {str(e)}",
                    data_source=data_source
                )
            raise e
    
    return wrapper


class EnhancedFileOperations:
    """Enhanced file operations with proper error handling"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.error_handler = ErrorHandler(self.logger)
    
    def safe_read_csv(self, file_path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
        """Safely read CSV file with proper error handling"""
        file_path = Path(file_path)
        
        context = {
            'file_path': str(file_path),
            'operation': 'read_csv'
        }
        
        try:
            with self.error_handler.handle_errors('file_read', context):
                if not file_path.exists():
                    raise DataProcessingError(
                        f"File does not exist: {file_path}",
                        data_source=str(file_path)
                    )
                
                if file_path.stat().st_size == 0:
                    raise DataProcessingError(
                        f"File is empty: {file_path}",
                        data_source=str(file_path)
                    )
                
                df = pd.read_csv(file_path, **kwargs)
                
                self.logger.info(f"Successfully read CSV file: {len(df)} rows, {len(df.columns)} columns", 
                               extra=context)
                
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to read CSV file: {file_path}", extra=context)
            return None
    
    def safe_write_csv(self, df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
        """Safely write CSV file with proper error handling"""
        file_path = Path(file_path)
        
        context = {
            'file_path': str(file_path),
            'operation': 'write_csv',
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        try:
            with self.error_handler.handle_errors('file_write', context):
                # Create parent directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write CSV file
                df.to_csv(file_path, **kwargs)
                
                self.logger.info(f"Successfully wrote CSV file: {len(df)} rows, {len(df.columns)} columns", 
                               extra=context)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to write CSV file: {file_path}", extra=context)
            return False
    
    def safe_read_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Safely read JSON file with proper error handling"""
        file_path = Path(file_path)
        
        context = {
            'file_path': str(file_path),
            'operation': 'read_json'
        }
        
        try:
            with self.error_handler.handle_errors('file_read', context):
                if not file_path.exists():
                    raise DataProcessingError(
                        f"JSON file does not exist: {file_path}",
                        data_source=str(file_path)
                    )
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.logger.info(f"Successfully read JSON file: {len(data) if isinstance(data, (dict, list)) else 'unknown'} items", 
                               extra=context)
                
                return data
                
        except Exception as e:
            self.logger.error(f"Failed to read JSON file: {file_path}", extra=context)
            return None
    
    def safe_write_json(self, data: Any, file_path: Union[str, Path]) -> bool:
        """Safely write JSON file with proper error handling"""
        file_path = Path(file_path)
        
        context = {
            'file_path': str(file_path),
            'operation': 'write_json'
        }
        
        try:
            with self.error_handler.handle_errors('file_write', context):
                # Create parent directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
                
                self.logger.info(f"Successfully wrote JSON file", extra=context)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to write JSON file: {file_path}", extra=context)
            return False


def create_error_report(exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a comprehensive error report for troubleshooting"""
    context = context or {}
    
    error_report = {
        'timestamp': time.time(),
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': traceback.format_exc(),
        'context': context,
        'system_info': {
            'platform': 'python',
            'version': '3.9+'  # Assuming Python 3.9+
        }
    }
    
    # Add UFC-specific error information
    if isinstance(exception, UFCPredictorError):
        error_report.update({
            'error_code': exception.error_code,
            'ufc_context': exception.context,
            'ufc_timestamp': exception.timestamp
        })
    
    return error_report


# Global error handling utilities
file_ops = EnhancedFileOperations()

def setup_global_exception_handler():
    """Setup global exception handler for unhandled exceptions"""
    import sys
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_logger = get_logger('global_exception_handler')
        log_exception(error_logger, exc_value, {
            'type': exc_type.__name__,
            'traceback': ''.join(traceback.format_tb(exc_traceback))
        })
    
    sys.excepthook = handle_exception


# Initialize global exception handling
setup_global_exception_handler()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Enhanced Error Handling Demo")
    print("=" * 50)
    
    # Setup logging
    from .logging_config import setup_logging
    setup_logging(level='DEBUG')
    
    # Demo enhanced prediction wrapper
    @enhanced_prediction_wrapper
    def demo_prediction(fighter_a: str, fighter_b: str) -> dict:
        if fighter_a == "Error Fighter":
            raise ValueError("Demo error for testing")
        return {
            "predicted_winner": fighter_a,
            "confidence": "75%"
        }
    
    # Demo enhanced scraping wrapper
    @enhanced_scraping_wrapper
    def demo_scraping(url: str) -> list:
        if "bad" in url:
            raise requests.RequestException("Demo network error")
        return ["data1", "data2", "data3"]
    
    # Test successful operations
    try:
        result = demo_prediction("Jon Jones", "Stipe Miocic")
        print(f"✅ Prediction result: {result}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # Test error handling
    try:
        result = demo_prediction("Error Fighter", "Opponent")
        print(f"Result: {result}")
    except PredictionError as e:
        print(f"✅ Caught prediction error: {e.error_code}")
    
    # Test file operations
    file_ops = EnhancedFileOperations()
    test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    success = file_ops.safe_write_csv(test_data, "test_output.csv", index=False)
    print(f"✅ File write success: {success}")
    
    loaded_data = file_ops.safe_read_csv("test_output.csv")
    if loaded_data is not None:
        print(f"✅ File read success: {len(loaded_data)} rows")
    
    print("\n✅ Enhanced error handling demo completed")