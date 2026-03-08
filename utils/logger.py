"""Logging configuration for Math Mentor"""
import logging
import sys
from pathlib import Path
from utils.config import Config

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # File handler if not in debug mode
        if not Config.DEBUG:
            log_dir = Config.DATA_DIR / "logs"
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "math_mentor.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger
