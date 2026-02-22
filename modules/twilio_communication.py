import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from twilio.rest import Client

# Get module name dynamically
module_name = os.path.splitext(os.path.basename(__file__))[0]

# Configure logger for this module
logger = logging.getLogger(module_name)
logger.setLevel(logging.DEBUG)

# Create a file handler per module
os.makedirs("logs", exist_ok=True)
log_file = f"logs/{module_name}.log"
file_handler = RotatingFileHandler(log_file, maxBytes=100*1024*1024, backupCount=5)

# Create and set formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler if not already added
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    
logger.debug(f"Logger initialized for {module_name}")

load_dotenv()

def send_message(type: str, key: str):
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        client = Client(account_sid, auth_token)

        message = client.messages.create(
        from_=os.environ.get("TWILIO_FROM_"),
        content_sid=os.environ.get("TWILIO_CONTENT_SID"),
        content_variables=f'{{"1":"{type}","2":"{key}"}}',
        to=os.environ.get("TWLIO_TO", "")
        )
        print(message.sid)
    except Exception as e:
        logger.error("Error in TWILIO: %s", e)