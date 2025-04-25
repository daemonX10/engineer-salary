import logging
import sys
from config import LOG_FILE

def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'), # Append mode
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Suppress overly verbose logs from libraries
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pytorch_tabnet").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully.") 