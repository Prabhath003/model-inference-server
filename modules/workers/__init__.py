
import logging
import os
from logging.handlers import RotatingFileHandler

# Get module name dynamically
module_name = os.path.basename(os.path.dirname(__file__))

# Configure logger for this module
logger = logging.getLogger(module_name)
logger.setLevel(logging.DEBUG)

os.makedirs("logs", exist_ok=True)
    
# Define the log file path
log_file = f"logs/{module_name}.log"

# Create a rotating file handler for logging
file_handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=5)

# Create and set a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger if not already added
if not logger.hasHandlers():
    logger.addHandler(file_handler)

# Log the successful initialization of the logger
logger.debug(f"Logger initialized for {module_name}")

from .huggingface import CLIPWorker, ImageTextToTextWorker, SentTransWorker, TextGenerationBnbWorker, TextGenerationWorker
from .genai_worker import GenAIWorker
from .ollama_worker import OllamaWorker
from .openai_worker import OpenAIWorker

__all__ = [
    "CLIPWorker",
    "ImageTextToTextWorker",
    "SentTransWorker",
    "TextGenerationBnbWorker",
    "TextGenerationWorker",
    "OpenAIWorker",
    "GenAIWorker",
    "OllamaWorker"
]