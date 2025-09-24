"""
Logging utilities for the Vectara vector database.
"""

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration for the entire application.
    
    Args:
        config: Logging configuration dictionary
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "log_to_file": False,
        "log_file": "vectara.log",
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
    
    # Merge with provided config
    config = {**default_config, **config}
    
    # Get logging level
    level = getattr(logging, config["level"].upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        config["format"],
        datefmt=config["date_format"]
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if requested
    if config["log_to_file"]:
        log_file = Path(config["log_file"])
        
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config["max_file_size"],
            backupCount=config["backup_count"]
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics for a function.
    
    Args:
        func_name: Name of the function
        duration: Execution duration in seconds
        **kwargs: Additional metrics to log
    """
    logger = get_logger("vectara.performance")
    
    metrics = {
        "function": func_name,
        "duration": duration,
        **kwargs
    }
    
    logger.info(f"Performance: {metrics}")


def log_memory_usage(operation: str, memory_before: int, memory_after: int) -> None:
    """
    Log memory usage for an operation.
    
    Args:
        operation: Name of the operation
        memory_before: Memory usage before operation (bytes)
        memory_after: Memory usage after operation (bytes)
    """
    logger = get_logger("vectara.memory")
    
    memory_delta = memory_after - memory_before
    memory_delta_mb = memory_delta / (1024 * 1024)
    
    logger.info(f"Memory usage - {operation}: {memory_delta_mb:.2f} MB delta")


def log_vector_operation(operation: str, vector_count: int, vector_dim: int, **kwargs) -> None:
    """
    Log vector operation metrics.
    
    Args:
        operation: Name of the operation
        vector_count: Number of vectors processed
        vector_dim: Dimension of vectors
        **kwargs: Additional operation-specific metrics
    """
    logger = get_logger("vectara.operations")
    
    metrics = {
        "operation": operation,
        "vector_count": vector_count,
        "vector_dim": vector_dim,
        **kwargs
    }
    
    logger.info(f"Vector operation: {metrics}")


def log_index_stats(index_type: str, vector_count: int, **stats) -> None:
    """
    Log index statistics.
    
    Args:
        index_type: Type of index
        vector_count: Number of vectors in index
        **stats: Additional index statistics
    """
    logger = get_logger("vectara.index")
    
    metrics = {
        "index_type": index_type,
        "vector_count": vector_count,
        **stats
    }
    
    logger.info(f"Index stats: {metrics}")


def log_search_query(query_vector_dim: int, k: int, results_count: int, duration: float) -> None:
    """
    Log search query metrics.
    
    Args:
        query_vector_dim: Dimension of query vector
        k: Number of results requested
        results_count: Number of results returned
        duration: Search duration in seconds
    """
    logger = get_logger("vectara.search")
    
    metrics = {
        "query_vector_dim": query_vector_dim,
        "k": k,
        "results_count": results_count,
        "duration": duration
    }
    
    logger.info(f"Search query: {metrics}")


def log_error(error: Exception, context: Optional[str] = None) -> None:
    """
    Log an error with context.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger = get_logger("vectara.error")
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context
    }
    
    logger.error(f"Error occurred: {error_info}", exc_info=True)


def log_warning(message: str, **kwargs) -> None:
    """
    Log a warning message.
    
    Args:
        message: Warning message
        **kwargs: Additional context
    """
    logger = get_logger("vectara.warning")
    
    if kwargs:
        logger.warning(f"{message} - Context: {kwargs}")
    else:
        logger.warning(message)


def log_info(message: str, **kwargs) -> None:
    """
    Log an info message.
    
    Args:
        message: Info message
        **kwargs: Additional context
    """
    logger = get_logger("vectara.info")
    
    if kwargs:
        logger.info(f"{message} - Context: {kwargs}")
    else:
        logger.info(message)


def log_debug(message: str, **kwargs) -> None:
    """
    Log a debug message.
    
    Args:
        message: Debug message
        **kwargs: Additional context
    """
    logger = get_logger("vectara.debug")
    
    if kwargs:
        logger.debug(f"{message} - Context: {kwargs}")
    else:
        logger.debug(message)
