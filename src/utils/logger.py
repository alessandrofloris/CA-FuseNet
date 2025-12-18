import logging
import os
import sys

def setup_logger(log_dir="logs", log_filename="training.log", level=logging.INFO):
    """
    Sets up a custom logger that writes messages to both the console and a file.
    
    Parameters:
    - log_dir (str): Directory where the log file will be saved.
    - log_filename (str): Name of the log file.
    - level (int): Minimum logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
    - logging.Logger: The configured logger instance.
    """
    
    # Ensure the log directory exists
    log_path = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Get the root logger instance
    logger = logging.getLogger('CA-FuseNet-Logger')
    logger.setLevel(level)
    
    # Prevent duplicate handlers if the function is called multiple times
    if logger.handlers:
        return logger
        
    # 2. Define the formatter for message structure
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 3. File Handler: Writes logs to a file
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    
    # 4. Console Handler: Writes logs to the standard output (terminal)
    # Use StreamHandler with sys.stdout for consistency
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 5. Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global access function (best practice)
def get_logger():
    """Returns the globally configured logger instance."""
    return logging.getLogger('CA-FuseNet-Logger')