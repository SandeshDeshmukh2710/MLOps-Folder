import logging

def setup_logger(info_log_file: str, error_log_file: str) -> logging.Logger:
    """
    Set up a logger with separate files for INFO and ERROR levels.

    Args:
        info_log_file (str): Path to the log file for informational messages.
        error_log_file (str): Path to the log file for error messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("MLPipeline")
    logger.setLevel(logging.DEBUG)  # Log all levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # File handler for informational logs (INFO and DEBUG)
    info_handler = logging.FileHandler(info_log_file)
    info_handler.setLevel(logging.INFO)

    # File handler for error logs (ERROR and above)
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)

    # Console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger
