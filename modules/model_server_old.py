import multiprocessing
from flask import Flask, request, jsonify
from multiprocessing import Process, Manager
import threading
import psutil
import time
import hashlib
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import requests
from typing import Dict, Any, List, Tuple, Optional, Literal
import json
from transformers import pipeline, CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
import traceback
from vllm import LLM, SamplingParams
import base64
from io import BytesIO
from PIL import Image
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import random

from .utils import get_optimal_gpu_set, get_available_gpus
from .twilio_communication import send_message

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

API_KEYS = os.environ.get("GEMINI_API_KEYS", "").split(" ")
PORTS = [int(port) for port in os.environ.get("OLLAMA_PORTS", "").split(" ")]
OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", "").split(" ")

WORKER_TYPES = ["text-generation", "sent-trans", "image-text-to-text", "openai", "genai", "ollama", "clip", "claude", "openai-embedding"]

def call_genai(model_name: str, api_key: str, payload: Dict[str, Any]):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()['candidates'][0]['content']['parts'][0]['text']

def call_ollama(payload: Dict[str, Any], port: int):
    if port in [6001, 6002]:
        url = f"http://localhost:{port}/api/chat"
    else:
        raise ValueError(f"Invalid port: {port}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

# def call_openai(payload: Dict[str, Any], api_key: str):
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     response.raise_for_status()
#     return response.json()['choices'][0]['message']['content']

def call_openai(payload: Dict[str, Any], api_key: str):
    endpoint = "https://gika-openai.openai.azure.com"
    deployment = payload.get('model', 'gpt-4.1-mini')
    api_version = "2025-01-01-preview"
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def call_openai_batch( # only for chat completions
    payloads: List[Dict[str, Any]],
    azure_endpoint: str=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key: str=os.environ.get("AZURE_OPENAI_API_KEY", ""),
    api_version: str="2025-01-01-preview",
    batch_id: Optional[str]=None,
    poll_interval: int=10,
    completion_window: Literal["24h"]="24h"
):
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        # azure_deployment="gpt-4.1-nano-2",
        api_version=api_version,
        api_key=api_key
    )
    
    jsonl_lines = [
        json.dumps({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/chat/completions",
            "body": entry        
        }) for i, entry in enumerate(payloads)
    ]
    input_ids = {f"request_{i}": i for i in range(len(payloads))}
    outputs: List[Optional[Dict[str, Any]]] = [None] * len(input_ids)

    jsonl_content = "\n".join(jsonl_lines) + "\n"
    jsonl_bytes = jsonl_content.encode("utf-8")
    jsonl_file = BytesIO(jsonl_bytes)

    jsonl_file.seek(0)

    if not batch_id:
        response = client.files.create(
            file=("requests.jsonl", jsonl_file),
            purpose="batch"
        )
        file_id = response.id

        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions", # type:ignore
            completion_window=completion_window
        )
        batch_id = response.id
        
    print(batch_id)
    start_time = time.time()
    max_wait_time = 24*3600
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            return outputs
        response = client.batches.retrieve(batch_id)
        if response.status == 'completed':
            break
        elif response.status in ['failed', 'expired', 'cancelled']:
            return outputs
        print(response.status, end="\r")
        time.sleep(poll_interval)

    if response.output_file_id:
        file_content = client.files.content(response.output_file_id)
        text = file_content.content.decode("utf-8")
        results = [json.loads(line) for line in text.splitlines() if line.strip()]
        for result in results:
            if result and result['custom_id'] in input_ids:
                outputs[result['custom_id']] = {
                    "response": result['response']["body"]["choices"][-1]['message']['content'],
                    "usage": result['response']["body"]["usage"]
                }
    
    return outputs

def call_openai_embedding(payload: Dict[str, Any], api_key: str):
    """
    Call Azure OpenAI Embeddings API.

    Args:
        payload: Dictionary containing:
            - input: List of strings or single string to embed
            - model: Model name (e.g., "text-embedding-3-small")
        api_key: Azure OpenAI API key

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    endpoint = "https://gika-openai.openai.azure.com"
    deployment = payload.get("model", "text-embedding-3-small")
    api_version = "2023-05-15"
    url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    request_payload = {
        "input": payload["input"],
        "model": payload.get("model", "text-embedding-3-small")
    }

    response = requests.post(url, headers=headers, json=request_payload)
    response.raise_for_status()

    # Return list of embeddings
    return [item["embedding"] for item in response.json()["data"]]

def call_claude(
    payload: Dict[str, Any], 
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0", 
    region_name: str = "us-east-1",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Call Claude API via AWS Bedrock with optional prompt caching.

    Args:
        payload: Dictionary containing:
            - messages: List of message dictionaries with role and content
            - max_tokens: Optional maximum tokens (default: 4000)
            - temperature: Optional temperature (default: 0.1)
            - system: Optional system prompt (can include cache_control)
        model_id: Claude model ID (default: Sonnet 4)
        region_name: AWS region (default: us-east-1)
        use_cache: Enable prompt caching (default: False)

    Returns:
        dict: Response containing:
            - text: The response text from Claude
            - usage: Token usage statistics including cache metrics

    Raises:
        Exception: If API call fails
    """
    # Initialize Bedrock client
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name
    )

    # Extract parameters from payload
    messages = payload.get("messages", [])
    max_tokens = payload.get("max_tokens", 4000)
    temperature = payload.get("temperature", 0.1)
    system_prompt = payload.get("system", None)

    # Convert messages format if needed (from OpenAI format to Claude format)
    claude_messages = []
    for msg in messages:
        # Handle both simple string content and complex content arrays
        content = msg.get("content", "")
        cache_control = msg.get("cache_control", None)  # Check for cache control
        
        if isinstance(content, str):
            message_content = [{"text": content}]
            
            # Add cache control if specified and caching is enabled
            if use_cache and cache_control:
                message_content[0]["cacheControl"] = {"type": cache_control.get("type", "ephemeral")}
            
            claude_messages.append({
                "role": msg.get("role", "user"),
                "content": message_content
            })
            
        elif isinstance(content, list):
            # Already in the right format or needs conversion
            formatted_content = []
            for idx, item in enumerate(content):
                if isinstance(item, dict):
                    if "text" in item and "type" not in item:
                        # Already in correct Claude format (text only, no type field)
                        formatted_item = {"text": item["text"]}
                        
                        # Add cache control to the last content block if specified
                        if use_cache and idx == len(content) - 1 and cache_control:
                            formatted_item["cacheControl"] = {"type": cache_control.get("type", "ephemeral")}
                        
                        formatted_content.append(formatted_item)
                        
                    elif "type" in item:
                        item_type = item["type"]

                        if item_type == "text":
                            formatted_item = {"text": item.get("text", "")}
                            
                            # Add cache control to the last content block if specified
                            if use_cache and idx == len(content) - 1 and cache_control:
                                formatted_item["cacheControl"] = {"type": cache_control.get("type", "ephemeral")}
                            
                            formatted_content.append(formatted_item)

                        elif item_type == "document":
                            # Handle document (PDF) upload
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                formatted_item = {
                                    "document": {
                                        "format": "pdf",
                                        "name": "uploaded_document",
                                        "source": {
                                            "bytes": base64.b64decode(source.get("data", ""))
                                        }
                                    }
                                }
                                
                                # Add cache control for documents if specified
                                if use_cache and item.get("cache_control"):
                                    formatted_item["cacheControl"] = {"type": "ephemeral"}
                                
                                formatted_content.append(formatted_item)

                        elif item_type == "image" or item_type == "image_url":
                            # Handle image upload
                            if item_type == "image_url":
                                # For now, log that image URLs need conversion
                                logger.warning("Image URLs require conversion to base64 - skipping")
                            elif "source" in item:
                                source = item.get("source", {})
                                if source.get("type") == "base64":
                                    formatted_item = {
                                        "image": {
                                            "format": source.get("media_type", "image/png").split("/")[-1],
                                            "source": {
                                                "bytes": base64.b64decode(source.get("data", ""))
                                            }
                                        }
                                    }
                                    
                                    # Add cache control for images if specified
                                    if use_cache and item.get("cache_control"):
                                        formatted_item["cacheControl"] = {"type": "ephemeral"}
                                    
                                    formatted_content.append(formatted_item)

                elif isinstance(item, str):
                    formatted_content.append({"text": item})

            if formatted_content:
                claude_messages.append({
                    "role": msg.get("role", "user"),
                    "content": formatted_content
                })

    # Prepare converse parameters
    converse_params = {
        "modelId": model_id,
        "messages": claude_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }

    # Add system prompt with optional cache control
    if system_prompt:
        if isinstance(system_prompt, str):
            # Simple string system prompt
            system_blocks = [{"text": system_prompt}]
            
            # Add cache control to system prompt if caching is enabled
            if use_cache:
                system_blocks[0]["cacheControl"] = {"type": "ephemeral"}
            
            converse_params["system"] = system_blocks
            
        elif isinstance(system_prompt, list):
            # System prompt is already a list of blocks
            system_blocks = []
            for idx, block in enumerate(system_prompt):
                if isinstance(block, dict):
                    formatted_block = {"text": block.get("text", "")}
                    
                    # Add cache control to marked blocks or the last block
                    if use_cache:
                        if block.get("cache_control"):
                            formatted_block["cacheControl"] = {"type": "ephemeral"}
                        elif idx == len(system_prompt) - 1:
                            # Cache the last system block by default
                            formatted_block["cacheControl"] = {"type": "ephemeral"}
                    
                    system_blocks.append(formatted_block)
                elif isinstance(block, str):
                    system_blocks.append({"text": block})
            
            converse_params["system"] = system_blocks

    # Make the API call
    response = client.converse(**converse_params)

    # Extract text and usage from response
    if 'output' in response and 'message' in response['output']:
        content = response['output']['message'].get('content', [])
        if content and isinstance(content, list):
            return {
                "text": content[0].get('text', ''),
                "usage": response.get("usage", {})
            }

    raise Exception("Invalid response format from Claude API")

def to_pil_image(source) -> Image.Image:
    """
    Convert file path, URL, base64, bytes, or PIL image into a PIL.Image.Image (RGB).
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    
    if isinstance(source, (bytes, bytearray)):
        return Image.open(BytesIO(source)).convert("RGB")
    
    if isinstance(source, str):
        if os.path.exists(source):  # File path
            return Image.open(source).convert("RGB")
        
        if source.startswith("http://") or source.startswith("https://"):  # URL
            resp = requests.get(source)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        
        if source.startswith("data:image"):
            source = source.split(",")[1]  # Remove prefix
        
        # Base64 string
        try:
            img_bytes = base64.b64decode(source, validate=True)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not load image from string: {e}")
    
    raise TypeError(f"Unsupported image source type: {type(source)}")


class ModelServer:
    """
    A programmable search engine that processes search queries using different Selenium drivers.
    
    Attributes:
        max_retries (int): Maximum retries allowed for a failed query.
        max_requests (int): Maximum number of queries to process.
        driver_reset_threshold (int): Number of driver uses before restart.
        completed_queries (dict): Shared dictionary for storing completed query results.
        ready_queries (dict): Shared dictionary for pending queries.
        restart_event (Event): Event to trigger restart of query processing.
        new_query_event (Event): Signaling event for a new query.
        completed_query_event (Event): Signaling event when a query completes.
        drivers (dict): For caching drivers if needed.
    """
    
    def __init__(self):
        """
        Initialize the search engine with configuration parameters.
        Parameters:
        """
        try:
            logger.info("Initializing ModelServer...")
            logger.debug("Creating multiprocessing Manager")
            self.manager = Manager()

            logger.debug("Initializing shared dictionaries and locks")
            self.completed_requests = self.manager.dict()
            self._completedLock = threading.Lock()

            self.ready_requests = self.manager.dict()
            self._readyLock = threading.Lock()

            logger.debug("Initializing GPU locks")
            self._gpuLock = threading.Lock()
            self._gpuLockM = self.manager.Lock()

            logger.debug("Initializing worker management structures")
            self.workerRequests = self.manager.list()
            self.workerProcs = self.manager.dict()
            self._workerLock = threading.Lock()
            self._workerLockM = self.manager.Lock()
            self.new_worker_event = self.manager.Event()

            self.start_time = time.time()

            # Initialize persistence
            logger.debug("Initializing persistence")
            self.persistence_dir = "completed_requests_persistence"
            os.makedirs(self.persistence_dir, exist_ok=True)
            # self.load_completed_requests()
        except Exception as e:
            logger.error(f"Failed to initialize ModelServer: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_persistence_file(self, type: str, model_name: str) -> str:
        """Get the persistence file path for a specific type/model combination."""
        safe_name = f"{type}_{model_name}".replace("/", "_").replace("\\", "_")
        return os.path.join(self.persistence_dir, f"{safe_name}_completed.json")

    def save_completed_requests(self, type: str, model_name: str) -> None:
        """Save completed requests for a specific type/model to disk using atomic writes."""
        try:
            if type not in self.completed_requests or model_name not in self.completed_requests[type]:
                return

            file_path = self._get_persistence_file(type, model_name)
            # Filter out entries with errors - only save successful requests
            data = {uuid: entry for uuid, entry in self.completed_requests[type][model_name].items()
                    if "error" not in entry}

            # Use atomic write: write to temp file then rename
            temp_file_path = file_path + ".tmp"
            with open(temp_file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Atomic rename (on most filesystems)
            os.replace(temp_file_path, file_path)

            logger.debug(f"Saved {len(data)} completed requests for {type}/{model_name} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save completed requests for {type}/{model_name}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def load_completed_requests(self) -> None:
        """Load all completed requests from disk."""
        try:
            if not os.path.exists(self.persistence_dir):
                logger.debug("Persistence directory does not exist yet")
                return

            files = [f for f in os.listdir(self.persistence_dir) if f.endswith("_completed.json")]
            total_loaded = 0

            for file in files:
                try:
                    file_path = os.path.join(self.persistence_dir, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # Extract type and model_name from filename
                    name_parts = file.replace("_completed.json", "").rsplit("_", 1)
                    if len(name_parts) != 2:
                        logger.warning(f"Could not parse filename: {file}")
                        continue

                    type_name = name_parts[0]
                    model_name = name_parts[1]

                    # Initialize nested dicts if needed
                    if type_name not in self.completed_requests:
                        self.completed_requests[type_name] = self.manager.dict()
                    if model_name not in self.completed_requests[type_name]:
                        self.completed_requests[type_name][model_name] = self.manager.dict()

                    # Load data
                    for uuid, entry in data.items():
                        self.completed_requests[type_name][model_name][uuid] = entry

                    total_loaded += len(data)
                    logger.info(f"Loaded {len(data)} completed requests for {type_name}/{model_name} from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load completed requests from {file}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")

            logger.info(f"Total completed requests loaded from disk: {total_loaded}")
        except Exception as e:
            logger.error(f"Failed to load completed requests: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def start_flask_server(self):
        """
        Start the Flask server that listens for search queries.
        Documented endpoint: POST /search returns query results or error.
        """
        app = Flask(__name__)
        app.secret_key = "model-server"
        app.config["SESSION_TYPE"] = "filesystem"
        
        @app.route("/infer", methods=["POST"])
        def infer():
            try:
                logger.debug("Received POST request to /infer")
                data: Optional[Dict[str, Any]] = request.json
                if not data:
                    return jsonify({"error": "Missing Params"}), 400
                payload = data.get("payload")
                model_name = data.get("model_name")
                type = data.get("type")
                if not payload or not model_name or not type:
                    logger.error("Missing payload or model_name or type in request")
                    return jsonify({"error": "Missing payload or model_name or type"}), 400
                timeout: int = data.get("timeout", 300)
                max_retries: int = data.get("max_retries", 3)
                force: bool = data.get("force", False)

                self.check_and_create_worker(type, model_name, timeout)
                uuid: str = hashlib.md5(json.dumps(payload).encode("utf-8")).hexdigest()
                # uuid: str = str(uuid4())
                self.enqueue_request(type, model_name, uuid, payload, retry_count=0, max_retries=max_retries, force=force)
                response = self.get_response(type, model_name, uuid, timeout)
                if response:
                    return jsonify({"response": response}), 200
                error = "Failed to get response"
                return jsonify({"error": error}), 500
            except Exception as e:
                logger.error(f"Exception in /infer endpoint: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
        logger.info("Starting Flask server on port 1121")
        app.run(host="0.0.0.0", port=1121)
        
    def check_and_create_worker(self, type: str, model_name: str, timeout: int=300):
        with self._workerLock, self._workerLockM:
            if not (type in self.workerProcs and model_name in self.workerProcs[type]):
                self.workerRequests.append((type, model_name))
                self.new_worker_event.set()
        start_time = time.time()
        if timeout == 0:
            while True:
                with self._workerLock, self._workerLockM:
                    if type in self.workerProcs and model_name in self.workerProcs[type]:
                        return
        else:
            while time.time() - start_time < timeout:
                with self._workerLock, self._workerLockM:
                    if type in self.workerProcs and model_name in self.workerProcs[type]:
                        return
        raise ValueError("Cannot able to serve the requested type")
            
    def dequeue_worker_request(self) -> Optional[Tuple[str, str]]:
        with self._workerLock, self._workerLockM:
            if self.workerRequests:
                return self.workerRequests.pop(0)
        return None
        
    def create_worker(self):
        while True:
            request = self.dequeue_worker_request()
            if request:
                type, model_name = request
                if type not in self.ready_requests:
                    self.ready_requests[type] = self.manager.dict()
                if model_name not in self.ready_requests[type]:
                    self.ready_requests[type][model_name] = self.manager.dict()
                if type not in self.completed_requests:
                    self.completed_requests[type] = self.manager.dict()
                if model_name not in self.completed_requests[type]:
                    self.completed_requests[type][model_name] = self.manager.dict()
                with self._workerLockM:
                    if type not in self.workerProcs:
                        self.workerProcs[type] = self.manager.dict()
                    if model_name not in self.workerProcs[type]:
                        self.workerProcs[type][model_name] = self.manager.dict()
                        
                        self.workerProcs[type][model_name]["type"] = type
                        self.workerProcs[type][model_name]["model_name"] = model_name
                        self.workerProcs[type][model_name]["readylockM"] = self.manager.Lock()
                        self.workerProcs[type][model_name]["new_request_event"] = self.manager.Event()
                        self.workerProcs[type][model_name]["completedlockM"] = self.manager.Lock()
                        self.workerProcs[type][model_name]["completed_request_event"] = self.manager.Event()
                        
                        logger.info(f"Created worker for {type} - {model_name}: {self.workerProcs[type][model_name]}")
                        
                        # Store config for this worker
                        worker_config: Dict[str, Any] = {
                            "model_name": model_name,
                            "type": type,
                            # "max_workers": 1 if model_name in ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"] else 1
                            "max_workers": 8
                        }
                        
                        # Create a separate process for each worker type
                        if type == "text-generation":
                            p = Process(target=self.text_generation_start_processor, kwargs=worker_config)
                            logger.info(f"Created text-generation worker for {model_name}")
                        elif type == "image-text-to-text":
                            p = Process(target=self.imageTextToText_start_processor, kwargs=worker_config)
                            logger.info(f"Created image-text-to-text worker for {model_name}")
                        elif type == "sent-trans":
                            p = Process(target=self.sent_trans_start_processor, kwargs=worker_config)
                            logger.info(f"Created sent-trans worker for {model_name}")
                        elif type == "clip":
                            p = Process(target=self.clip_generation_start_processor, kwargs=worker_config)
                            logger.info(f"Created CLIP worker for {model_name}")
                        elif type == "genai":
                            p = Process(target=self.genai_start_processor, kwargs=worker_config)
                            logger.info(f"Created genai worker for {model_name}")
                        elif type == "ollama":
                            p = Process(target=self.ollama_start_processor, kwargs=worker_config)
                            logger.info(f"Created ollama worker for {model_name}")
                        elif type == "openai":
                            p = Process(target=self.openai_start_processor, kwargs=worker_config)
                            logger.info(f"Created openai worker for {model_name}")
                        elif type == "vllm-text-generation":
                            p = Process(target=self.vllm_text_generation_start_processor, kwargs=worker_config)
                            logger.info(f"Created vllm-text-generation worker for {model_name}")
                        elif type == "claude":
                            p = Process(target=self.claude_start_processor, kwargs=worker_config)
                            logger.info(f"Created claude worker for {model_name}")
                        elif type == "openai-embedding":
                            p = Process(target=self.openai_embedding_start_processor, kwargs=worker_config)
                            logger.info(f"Created openai-embedding worker for {model_name}")
                        else:
                            logger.error(f"Unsupported model type: {type}")
                            continue
                        
                        # self.workerProcs[type][model_name]["process"] = p
                        p.start()
            else:
                self.new_worker_event.wait()
                self.new_worker_event.clear()
        
    def printStats(self):
        """
        Periodically log the current status of completed and remaining queries.
        """
        while True:
            time.sleep(60)
            logger.info(f"Memory usage: {psutil.virtual_memory().percent:.2f}%")
            logger.info(f"GPU usage: {get_available_gpus()}")
            logger.info(f"CPU usage: {psutil.cpu_percent(interval=1):.2f}%")
            if self.ready_requests:
                worker_types = list(self.ready_requests.keys())
                for worker_type in worker_types:
                    if self.ready_requests[worker_type]:
                        model_names = list(self.ready_requests[worker_type].keys())
                        for model_name in model_names:
                            logger.info(f"Total requests: {len(self.ready_requests[worker_type][model_name])} in {model_name} of {worker_type}")
                            logger.info(f"Completed requests: {len(self.completed_requests[worker_type][model_name])} in {model_name} of {worker_type}")
                            logger.info(f"Average response time: {(sum(response['response_time'] for response in self.completed_requests[worker_type][model_name].values()) / len(self.completed_requests[worker_type][model_name])) if self.completed_requests[worker_type][model_name] else 0:.2f} seconds in {model_name} of {worker_type}")
            
    def clean_complete_requests(self, only_clean_errors: bool = False):
        """
        Clean up completed queries that are older than one hour.

        Args:
            only_clean_errors: If True, only remove entries with errors. If False, remove all old entries.
        """
        while True:
            try:
                if self.completed_requests:
                    worker_types = list(self.completed_requests.keys())
                    for worker_type in worker_types:
                        if self.completed_requests[worker_type]:
                            model_names = list(self.completed_requests[worker_type].keys())
                            for model_name in model_names:
                                # Check if worker still exists
                                if worker_type not in self.workerProcs or model_name not in self.workerProcs[worker_type]:
                                    logger.debug(f"Worker {worker_type}/{model_name} not found, skipping cleanup")
                                    continue

                                uuids = list(self.completed_requests[worker_type][model_name].keys())
                                for uuid in uuids:
                                    with self._completedLock, self.workerProcs[worker_type][model_name]["completedlockM"]:
                                        entry = self.completed_requests[worker_type][model_name][uuid]
                                        if time.time() - entry["timestamp"] > 3600:
                                            if only_clean_errors and "error" not in entry:
                                                continue
                                            self.completed_requests[worker_type][model_name].pop(uuid)
            except Exception as e:
                logger.error(f"error in cleaning completed requests: {e}", exc_info=True)
            time.sleep(3600)
        
    def start(self):
        """
        Start processing: launch helper threads and processes, including the Flask server and driver processors.
        """
        logger.info("Starting ModelServer")
        stats_thread = threading.Thread(target=self.printStats, daemon=True)
        stats_thread.start()
        cleaner_thread = threading.Thread(target=self.clean_complete_requests, args=(False,), daemon=True)
        cleaner_thread.start()
        flask_server = Process(target=self.start_flask_server)
        flask_server.start()
        create_server = Process(target=self.create_worker)
        create_server.start()
        
        flask_server.join()
        stats_thread.join()
        create_server.join()
        cleaner_thread.join()
    
    def enqueue_request(self, type: str, model_name: str, uuid: str, payload: Dict[str, Any], retry_count: int = 0, max_retries: int = 3, force: bool = False):
        """
        Enqueue a new search query for processing.
        Parameters:
            uuid (str): Unique identifier for the query.
            payload (Dict[str, Any]): The request payload.
            retry_count (int): Current retry count.
            max_retries (int): Maximum number of retries allowed.
            force (bool): If True, remove from completed and re-enqueue. If False, skip if already completed.
        """
        logger.debug(f"Enqueueing request {uuid} for {type}/{model_name} (retry: {retry_count}/{max_retries}, force: {force})")

        # Check if max retries exceeded
        # if retry_count > max_retries:
        #     logger.error(f"Request {uuid} exceeded max retries ({max_retries}), discarding")
        #     with self._completedLock, self.workerProcs[type][model_name]["completedlockM"]:
        #         self.completed_requests[type][model_name][uuid] = {
        #             "timestamp": time.time(),
        #             "response_time": 0,
        #             "payload": payload,
        #             "error": f"Max retries ({max_retries}) exceeded",
        #             "retry_count": retry_count
        #         }
        #         self.workerProcs[type][model_name]["completed_request_event"].set()
        #     return

        # Check if already completed
        logger.debug(f"Checking if request {uuid} is already completed")
        if uuid in self.completed_requests[type][model_name]:
            if force:
                logger.info(f"Request {uuid} already completed, removing from completed and re-enqueueing (force=True)")
                with self._completedLock, self.workerProcs[type][model_name]["completedlockM"]:
                    self.completed_requests[type][model_name].pop(uuid)
                # Fall through to add to ready queue
            else:
                logger.info(f"Request {uuid} already completed, updating timestamp")
                with self._completedLock, self.workerProcs[type][model_name]["completedlockM"]:
                    self.completed_requests[type][model_name][uuid]["timestamp"] = time.time()
                    self.workerProcs[type][model_name]["completed_request_event"].set()
                return

        # Add to ready queue
        logger.debug(f"Adding request {uuid} to ready queue")
        with self._readyLock, self.workerProcs[type][model_name]["readylockM"]:
            self.ready_requests[type][model_name][uuid] = {
                "payload": payload,
                "timestamp": time.time(),
                "retry_count": retry_count,
                "max_retries": max_retries
            }

            queue_size = len(self.ready_requests[type][model_name])
            logger.info(f"Request {uuid} enqueued. Queue size for {type}/{model_name}: {queue_size}")
            self.workerProcs[type][model_name]["new_request_event"].set()
            
    def dequeue_request(self, type: str, model_name: str, count: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Dequeue the next available query from the pending queue.
        Returns:
            tuple: (uuid, query_info) or (None, None) if no query exists.
        """
        logger.debug(f"Attempting to dequeue {count} requests for {type}/{model_name}")
        with self.workerProcs[type][model_name]["readylockM"]:
            if self.ready_requests[type][model_name]:
                requests = list(self.ready_requests[type][model_name].keys())[:count]
                return [(key, self.ready_requests[type][model_name].pop(key)) for key in requests]
        logger.debug(f"Attempting to dequeue {count} requests for {type}/{model_name}")
        return []
    
    def get_response(self, type: str, model_name: str, uuid: str, timeout: int=300):
        logger.debug(f"Waiting for response for request {uuid} (timeout: {timeout}s)")
        start_time = time.time()
        if timeout == 0:
            while True:
                self.workerProcs[type][model_name]["completed_request_event"].wait()
                with self._completedLock, self.workerProcs[type][model_name]["completedlockM"]:
                    self.workerProcs[type][model_name]["completed_request_event"].clear()
                    if uuid in self.completed_requests[type][model_name]:
                        response_time = time.time() - start_time
                        logger.info(f"Got response for {uuid} after {response_time:.2f}s")
                        result = self.completed_requests[type][model_name][uuid]
                        if "error" in result:
                            logger.error(f"{result['error']}")
                            return None
                        return result.get("response")
                elapsed = time.time() - start_time
                if elapsed > 30 and int(elapsed) % 30 == 0:  # Log every 30 seconds
                    logger.warning(f"Still waiting for response for {uuid} ({elapsed:.0f}s elapsed)")
        else:
            while time.time() - start_time < timeout:
                self.workerProcs[type][model_name]["completed_request_event"].wait()
                with self._completedLock, self.workerProcs[type][model_name]["completedlockM"]:
                    self.workerProcs[type][model_name]["completed_request_event"].clear()
                    if uuid in self.completed_requests[type][model_name]:
                        response_time = time.time() - start_time
                        logger.info(f"Got response for {uuid} after {response_time:.2f}s")
                        result = self.completed_requests[type][model_name][uuid]
                        if "error" in result:
                            logger.error(f"{result['error']}")
                            return None
                        return result.get("response")
                elapsed = time.time() - start_time
                if elapsed > 30 and int(elapsed) % 30 == 0:  # Log every 30 seconds
                    logger.warning(f"Still waiting for response for {uuid} ({elapsed:.0f}s elapsed)")
        return None
    
    def add_to_completed(self, type: str, model_name: str, uuid: str, entry: Dict[str, Any], **kwargs: Any) -> None:
        logger.debug(f"Adding completed request {uuid} for {type}/{model_name}")
        with self.workerProcs[type][model_name]["completedlockM"]:
            response_time = time.time() - entry["timestamp"]
            self.completed_requests[type][model_name][uuid] = {
                "timestamp": entry["timestamp"],
                "response_time": response_time,
                "payload": entry["payload"],
                **kwargs
            }

            completed_count = len(self.completed_requests[type][model_name])
            logger.info(f"Request {uuid} completed in {response_time:.2f}s. Total completed for {type}/{model_name}: {completed_count}")

            self.workerProcs[type][model_name]["completed_request_event"].set()

            # Persist completed requests to disk
            # self.save_completed_requests(type, model_name)
            
    def openai_start_processor(self, **kwargs: Any):
        logger.info(f"Starting openai processor with config: {kwargs}")
            
        processes: List[Process] = []
        for api_key in OPENAI_API_KEYS:
            process = Process(target=self.openai_Subprocessor, args=(api_key,), kwargs=kwargs)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    
    def openai_Subprocessor(self, api_key: str, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str= kwargs.get("model_name", "")
        logger.info(f"Starting openai worker with config: {kwargs}  of api key: {api_key[:3]}...")
        # max_workers: int = kwargs.get("max_workers", 10)
        max_workers = max(1, int(0.2*(multiprocessing.cpu_count())))
        workers: List[Process] = []  
        shared_data: Dict[str, Any] = {
            'start_time': self.manager.Value('d', time.time()),
            'minute_start_time': self.manager.Value('d', time.time()),
            'total_requests_count': self.manager.Value('i', 0),
            'minute_requests_count': self.manager.Value('i', 0),
            'counts_lock': self.manager.Lock()
        }
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active openai subworkers: {len(workers)} of api key: {api_key[:3]}...")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new openai subworker (total will be: {len(workers) + 1}) of api key: {api_key[:3]}...")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.openaiWorker, args=(api_key, shared_data,), kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()
    
    def openai_embedding_start_processor(self, **kwargs: Any):
        logger.info(f"Starting openai embedding processor with config: {kwargs}")

        processes: List[Process] = []
        for api_key in OPENAI_API_KEYS:
            process = Process(target=self.openai_embedding_Subprocessor, args=(api_key,), kwargs=kwargs)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    def openai_embedding_Subprocessor(self, api_key: str, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str= kwargs.get("model_name", "")
        logger.info(f"Starting openai embedding worker with config: {kwargs} of api key: {api_key[:3]}...")
        max_workers = max(1, int(0.2*(multiprocessing.cpu_count())))
        workers: List[Process] = []
        shared_data: Dict[str, Any] = {
            'start_time': self.manager.Value('d', time.time()),
            'minute_start_time': self.manager.Value('d', time.time()),
            'total_requests_count': self.manager.Value('i', 0),
            'minute_requests_count': self.manager.Value('i', 0),
            'counts_lock': self.manager.Lock()
        }
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active openai embedding subworkers: {len(workers)} of api key: {api_key[:3]}...")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new openai embedding subworker (total will be: {len(workers) + 1}) of api key: {api_key[:3]}...")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.openaiEmbeddingWorker, args=(api_key, shared_data,), kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()

    def ollama_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str= kwargs.get("model_name", "")
        def worker(port: int):
            max_workers: int = kwargs.get("max_workers", 10)
            workers: List[Process] = []
            config: Dict[str, Any] = {
                'port': port,
                **kwargs
            }
            while True:
                workers = [w for w in workers if w.is_alive()]
                queue_length = len(self.ready_requests[type][model_name])
                ideal_workers = min(max_workers, (queue_length + 9) // 10)
                
                if len(workers) < ideal_workers:
                    for _ in range(ideal_workers - len(workers)):
                        p = Process(target=self.ollamaWorker, kwargs=config, daemon=True)
                        p.start()
                        workers.append(p)
                self.workerProcs[type][model_name]["new_request_event"].wait()
                with self.workerProcs[type][model_name]["readylockM"]:
                    self.workerProcs[type][model_name]["new_request_event"].clear()
            
        processes: List[Process] = []
        for port in PORTS:
            process = Process(target=worker, args=(port,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            
    def genai_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        def worker(api_key: str, shared_data: Dict[str, Any]):
            max_workers: int = kwargs.get("max_workers", 10)
            workers: List[Process] = []  
            config: Dict[str, Any] = {
                'api_key': api_key,
                'shared_data': shared_data,
                **kwargs
            }
            while True:
                workers = [w for w in workers if w.is_alive()]
                queue_length = len(self.ready_requests[type][model_name])
                ideal_workers = min(max_workers, (queue_length + 9) // 10)
                
                if len(workers) < ideal_workers:
                    for _ in range(ideal_workers - len(workers)):
                        p = Process(target=self.genaiWorker, kwargs=config, daemon=True)
                        p.start()
                        workers.append(p)
                self.workerProcs[type][model_name]["new_request_event"].wait()
                with self.workerProcs[type][model_name]["readylockM"]:
                    self.workerProcs[type][model_name]["new_request_event"].clear()
            
        processes: List[Process] = []
        for api_key in API_KEYS:
            process = Process(target=worker, args=(api_key, {
                'start_time': self.manager.Value('d', time.time()),
                'minute_start_time': self.manager.Value('d', time.time()),
                'total_requests_count': self.manager.Value('i', 0),
                'minute_requests_count': self.manager.Value('i', 0),
                'counts_lock': self.manager.Lock()
            },))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    
    def claude_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting claude processor with config: {kwargs}")
        max_workers = max(1, int(0.2*(multiprocessing.cpu_count())))
        workers: List[Process] = []

        # Shared throttling state across all workers
        shared_data: Dict[str, Any] = {
            'consecutive_throttles': self.manager.Value('i', 0),
            'last_throttle_time': self.manager.Value('d', 0),
            'throttle_lock': self.manager.Lock()
        }

        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active claude workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)

            if len(workers) < ideal_workers:
                logger.info(f"Spawning new claude worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    worker_config = {**kwargs, 'shared_data': shared_data}
                    p = Process(target=self.claudeWorker, kwargs=worker_config, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()

    def text_generation_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting text generation processor with config: {kwargs}")
        max_workers: int = kwargs.get("max_workers", 10)
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active text generation workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new text generation worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.textGenerationWorker, kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()

    def vllm_text_generation_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting vllm text generation processor with config: {kwargs}")
        max_workers: int = kwargs.get("max_workers", 3)
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active vllm text generation workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new vllm text generation worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.vllmTextGenerationWorker, kwargs=kwargs)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()
            
    def sent_trans_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting sen trans processor with config: {kwargs}")
        max_workers: int = kwargs.get("max_workers", 10)
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active sent trans workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new sent trans worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.sentTransWorker, kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()
            
    def clip_generation_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting clip processor with config: {kwargs}")
        max_workers: int = kwargs.get("max_workers", 10)
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active clip workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new clip worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.CLIPWorker, kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()
    
    def imageTextToText_start_processor(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting image text to text processor with config: {kwargs}")
        max_workers: int = kwargs.get("max_workers", 10)
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            logger.debug(f"Active image text to text workers: {len(workers)}")
            queue_length = len(self.ready_requests[type][model_name])
            ideal_workers = min(max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                logger.info(f"Spawning new image text to text worker (total will be: {len(workers) + 1})")
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.imageTextToTextWorker, kwargs=kwargs, daemon=True)
                    p.start()
                    workers.append(p)
            self.workerProcs[type][model_name]["new_request_event"].wait()
            with self.workerProcs[type][model_name]["readylockM"]:
                self.workerProcs[type][model_name]["new_request_event"].clear()

    def openaiWorker(self, api_key: str, shared_data: Dict[str, Any], **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        if not api_key or not shared_data:
            return
        logger.info(f"Starting openai worker for {type}/{model_name}")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()
        while True:
            now = time.time()
            if self.ready_requests[type][model_name]:
                with shared_data['counts_lock']:
                    # # if now - shared_data['start_time'].value > 86400:
                    # #     send_message("openai", "daily_limit resetted")
                    # #     shared_data['total_requests_count'].value = 0
                    # #     shared_data['start_time'].value = now
                    # if now - shared_data['minute_start_time'].value > 60:
                    #     shared_data['minute_requests_count'].value = 0
                    #     shared_data['minute_start_time'].value = now

                    # # if (shared_data['minute_requests_count'].value < 500 and
                    # #         shared_data['total_requests_count'].value < 10000):
                    # if (shared_data['minute_requests_count'].value < 5000):
                    #     requests = self.dequeue_request(type, model_name)
                    #     if requests:
                    #         uuid, entry = requests[0]
                    #         shared_data['minute_requests_count'].value += 1
                    #         # shared_data['total_requests_count'].value += 1
                    #     else:
                    #         uuid, entry = None, None
                    # # else:
                    # #     if shared_data['total_requests_count'].value >= 10000:
                    # #         send_message("openai", "crossed_daily_limit")
                    # #     uuid = None
                    # #     entry = None
                    requests = self.dequeue_request(type, model_name)
                    if requests:
                        uuid, entry = requests[0]
                    else:
                        uuid, entry = None, None
            else:
                uuid = None
                entry = None

            if uuid and entry:
                last_active = now
                try:
                    response = call_openai(entry["payload"], api_key)
                    self.add_to_completed(type, model_name, uuid, entry, response=response)
                except Exception as e:
                    logger.error("Error occurred in %s: %s", api_key, e)
                    retry_count = entry.get("retry_count", 0) + 1
                    max_retries = entry.get("max_retries", 3)
                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                    if "429" in str(e):
                        time.sleep(600)
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    break
    
    def openaiEmbeddingWorker(self, api_key: str, shared_data: Dict[str, Any], **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        if not api_key or not shared_data:
            return
        logger.info(f"Starting openai embedding worker for {type}/{model_name}")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()
        while True:
            now = time.time()
            if self.ready_requests[type][model_name]:
                with shared_data['counts_lock']:
                    # if now - shared_data['minute_start_time'].value > 60:
                    #     shared_data['minute_requests_count'].value = 0
                    #     shared_data['minute_start_time'].value = now

                    # if (shared_data['minute_requests_count'].value < 60000):
                    #     requests = self.dequeue_request(type, model_name)
                    #     if requests:
                    #         uuid, entry = requests[0]
                    #         shared_data['minute_requests_count'].value += 1
                    #     else:
                    #         uuid, entry = None, None
                    # else:
                    #     uuid = None
                    #     entry = None
                    raw_requests = self.dequeue_request(type, model_name, 1024)
                    grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
                    
                    for uuid, entry in raw_requests:
                        payload = entry["payload"]
                        sentences: List[str]|str = payload["input"]

                        # Handle pre-batched requests: messages is already a list of messages
                        if isinstance(sentences, list):
                            selected_batch = [(uuid, entry)]
                            # Put back all others
                            for _uuid, _entry in raw_requests:
                                if _uuid != uuid:
                                    retry_count = _entry.get("retry_count", 0)
                                    self.enqueue_request(type, model_name, _uuid, _entry["payload"], retry_count=retry_count)
                            break
                        # Exclude `messages` from the payload key for grouping
                        key = frozenset((k, v) for k, v in payload.items() if k != "input")
                        grouped[key].append((uuid, entry))
                    else:
                        # If not a pre-batched request, pick one uniform group (up to 8)
                        selected_batch = []
                        for batch in grouped.values():
                            selected_batch = batch[:32]
                            break

                        # Re-enqueue unused or extra requests
                        for batch in grouped.values():
                            if batch != selected_batch:
                                for uuid, entry in batch:
                                    retry_count = entry.get("retry_count", 0)
                                    max_retries = entry.get("max_retries", 3)
                                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                            elif len(batch) > 32:
                                for uuid, entry in batch[32:]:
                                    retry_count = entry.get("retry_count", 0)
                                    max_retries = entry.get("max_retries", 3)
                                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                    
            else:
                selected_batch = []
                
            if selected_batch:
                last_active = now
                try:
                    # Prepare batched input
                    input_texts: List[str] = []
                    response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                    for uuid, entry in selected_batch:
                        sentences: str | List[str] = entry["payload"]["input"]
                        if isinstance(sentences, list):
                            input_texts.extend(sentences)
                            response_counts.append((uuid, entry, len(sentences)))
                        else:
                            input_texts.append(sentences)
                            response_counts.append((uuid, entry, 1))
                    
                    # Get generation parameters
                    config = {k:v for k, v in selected_batch[0][1]["payload"].items() if k != "input"}
                    payload: Dict[str, Any] = {
                        **config,
                        "input": input_texts
                    }
                    
                    responses = call_openai_embedding(payload, api_key)
                    
                    # Assign responses correctly
                    index = 0
                    for uuid, entry, count in response_counts:
                        outputs = responses[index: index + count]
                        generated_texts: List[List[float]] = [o for o in outputs]
                        index += count

                        # If multiple outputs, store the list; otherwise a single response
                        final_response: List[float]|List[List[float]] = generated_texts if count > 1 or isinstance(entry["payload"]["input"], list) else generated_texts[0]

                        self.add_to_completed(type, model_name, uuid, entry, response=final_response)
                except Exception as e:
                    logger.error("Error occurred in %s: %s", model_name, e)
                    for uuid, entry in selected_batch:
                        retry_count = entry.get("retry_count", 0) + 1
                        self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                    time.sleep(60)

            # if uuid and entry:
            #     last_active = now
            #     try:
            #         response = call_openai_embedding(entry["payload"], api_key)
            #         self.add_to_completed(type, model_name, uuid, entry, response=response)
            #     except Exception as e:
            #         logger.error("Error occurred in openai embedding %s: %s", api_key, e)
            #         retry_count = entry.get("retry_count", 0) + 1
            #         max_retries = entry.get("max_retries", 3)
            #         self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
            #         if "429" in str(e):
            #             time.sleep(600)
            #         time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    break

    def ollamaWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        port: int = kwargs.get("port", "")
        if not port:
            return
        last_active = time.time()
        while True:
            now = time.time()
            if self.ready_requests[type][model_name]:
                requests = self.dequeue_request(type, model_name)
            else:
                requests = None
            if requests:
                uuid, entry = requests[0]
                last_active = now
                try:
                    response = call_ollama(entry["payload"], port)
                    self.add_to_completed(type, model_name, uuid, entry, response=response)
                except Exception as e:
                    logger.error("Error occurred in %s: %s", port, e)
                    retry_count = entry.get("retry_count", 0) + 1
                    max_retries = entry.get("max_retries", 3)
                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    break

    def genaiWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        api_key: str = kwargs.get("api_key", "")
        shared_data: Dict[str, Any] = kwargs.get("shared_data", {})
        if not api_key or not shared_data:
            return
        last_active = time.time()
        while True:
            now = time.time()
            if self.ready_requests[type][model_name]:
                with shared_data['counts_lock']:
                    if now - shared_data['start_time'].value > 86400:
                        send_message("genai", "daily_limit resetted")
                        shared_data['total_requests_count'].value = 0
                        shared_data['start_time'].value = now
                    if now - shared_data['minute_start_time'].value > 60:
                        shared_data['minute_requests_count'].value = 0
                        shared_data['minute_start_time'].value = now

                    if (shared_data['minute_requests_count'].value < 10 and
                            shared_data['total_requests_count'].value < 1500):
                        requests = self.dequeue_request(type, model_name)
                        if requests:
                            uuid, entry = requests[0]
                            shared_data['minute_requests_count'].value += 1
                            shared_data['total_requests_count'].value += 1
                        else:
                            uuid, entry = None, None
                    else:
                        if shared_data['total_requests_count'].value >= 1500:
                            send_message("genai", "crossed_daily_limit")
                        uuid = None
                        entry = None
            else:
                uuid = None
                entry = None

            if uuid and entry:
                last_active = now
                try:
                    response = call_genai(entry["model_name"], api_key, entry["payload"])
                    self.add_to_completed(type, model_name, uuid, entry, response=response)
                    time.sleep(5)
                except Exception as e:
                    logger.error("Error occurred in %s: %s", api_key, e)
                    retry_count = entry.get("retry_count", 0) + 1
                    max_retries = entry.get("max_retries", 3)
                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                    if "429" in str(e):
                        time.sleep(600)
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    break

    def claudeWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        region_name: str = kwargs.get("region_name", "us-east-1")
        shared_data: Dict[str, Any] = kwargs.get("shared_data", {})

        # Create fallback local throttling if no shared_data (shouldn't happen but be defensive)
        has_shared_data = bool(shared_data)
        if not has_shared_data:
            logger.warning(f"Claude worker started without shared_data, using local throttling state")
            # Use local state as fallback
            local_throttles = 0
            local_last_throttle = 0

        logger.info(f"Starting claude worker for {type}/{model_name} (shared_data: {has_shared_data})")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()

        while True:
            now = time.time()

            # Reset throttle counter if no throttling for 60 seconds
            if has_shared_data:
                with shared_data['throttle_lock']:
                    if now - shared_data['last_throttle_time'].value > 60:
                        if shared_data['consecutive_throttles'].value > 0:
                            logger.info(f"Resetting shared throttle backoff after 60s of no throttling")
                        shared_data['consecutive_throttles'].value = 0
            else:
                if now - local_last_throttle > 60:
                    if local_throttles > 0:
                        logger.info(f"Resetting local throttle backoff after 60s of no throttling")
                    local_throttles = 0

            if self.ready_requests[type][model_name]:
                requests = self.dequeue_request(type, model_name)
            else:
                requests = None

            if requests:
                uuid, entry = requests[0]
                last_active = now

                # Check if we should wait before making request due to throttling
                delay = 0
                current_throttles = 0

                if has_shared_data:
                    with shared_data['throttle_lock']:
                        current_throttles = shared_data['consecutive_throttles'].value
                        if current_throttles > 0:
                            delay = (2 ** min(current_throttles - 1, 5)) + random.uniform(0, 1)
                            logger.info(f"Pre-emptive backoff due to shared throttling (count: {current_throttles}), waiting {delay:.2f}s before request")
                else:
                    current_throttles = local_throttles
                    if current_throttles > 0:
                        delay = (2 ** min(current_throttles - 1, 5)) + random.uniform(0, 1)
                        logger.info(f"Pre-emptive backoff due to local throttling (count: {current_throttles}), waiting {delay:.2f}s before request")

                # Wait outside the lock to allow other workers to check
                if delay > 0:
                    time.sleep(delay)

                try:
                    response = call_claude(entry["payload"], model_id=model_name, region_name=region_name)
                    self.add_to_completed(type, model_name, uuid, entry, response=response)

                    # Reset throttle counter on success
                    if has_shared_data:
                        with shared_data['throttle_lock']:
                            if shared_data['consecutive_throttles'].value > 0:
                                logger.info(f"Request succeeded, resetting shared throttle counter from {shared_data['consecutive_throttles'].value} to 0")
                            shared_data['consecutive_throttles'].value = 0
                    else:
                        if local_throttles > 0:
                            logger.info(f"Request succeeded, resetting local throttle counter from {local_throttles} to 0")
                        local_throttles = 0

                except (BotoCoreError, ClientError) as e:
                    error_code = ''
                    if hasattr(e, 'response') and e.response:
                        error_code = e.response.get('Error', {}).get('Code', '')
                    error_message = str(e)

                    # Check if it's a throttling error
                    is_throttling = (
                        error_code == 'ThrottlingException' or
                        'Too many requests' in error_message or
                        'throttling' in error_message.lower() or
                        'rate limit' in error_message.lower()
                    )

                    logger.error(f"Claude API error ({error_code}): {error_message}")
                    retry_count = entry.get("retry_count", 0) + 1
                    max_retries = entry.get("max_retries", 3)
                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)

                    if is_throttling:
                        # Update throttling state
                        if has_shared_data:
                            with shared_data['throttle_lock']:
                                shared_data['consecutive_throttles'].value += 1
                                shared_data['last_throttle_time'].value = now
                                current_throttles = shared_data['consecutive_throttles'].value
                        else:
                            local_throttles += 1
                            local_last_throttle = now
                            current_throttles = local_throttles

                        # Calculate backoff
                        delay = (2 ** min(current_throttles - 1, 5)) + random.uniform(0, 1)
                        logger.warning(f"Throttling detected (consecutive: {current_throttles}), waiting {delay:.2f}s before retry")
                        time.sleep(delay)
                    else:
                        time.sleep(60)

                except Exception as e:
                    logger.error(f"Error occurred in Claude worker: {str(e)}")
                    retry_count = entry.get("retry_count", 0) + 1
                    max_retries = entry.get("max_retries", 3)
                    self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    break

    def textGenerationWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting text generation worker for {type}/{model_name}")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()
        pipe = None
        while True:
            now = time.time()
            if not pipe:
                if self.ready_requests[type][model_name]:
                    logger.info(f"Attempting to load model {model_name}")
                    try:
                        with self._gpuLock, self._gpuLockM:
                            available_gpus = get_optimal_gpu_set(model_name, dtype="fp16")
                            logger.debug(f"Available GPUs for {model_name}: {available_gpus}")
                            if available_gpus:
                                logger.info(f"Loading model {model_name} on GPU: {available_gpus[0]}")
                                start_load_time = time.time()

                                pipe = pipeline(
                                    "text-generation",
                                    model=model_name,
                                    device_map=f"cuda:{available_gpus[0]}",
                                    torch_dtype=torch.bfloat16,
                                    clean_up_tokenization_spaces=False,
                                    trust_remote_code=True
                                )

                                load_time = time.time() - start_load_time
                                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                            else:
                                logger.warning(f"No available GPUs for {model_name}, waiting...")
                                time.sleep(60)  # Wait before retry
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {str(e)}")
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        time.sleep(60)  # Wait before retry
                        continue

            if pipe and self.ready_requests[type][model_name]:
                logger.debug(f"Dequeuing up to 32 requests for {type}/{model_name}")
                raw_requests = self.dequeue_request(type, model_name, 32)
                grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

                for uuid, entry in raw_requests:
                    payload: Dict[str, Any] = entry["payload"]
                    inputs: List[Dict[str, str]] | List[List[Dict[str, str]]] = payload["inputs"]

                    # Handle pre-batched requests: inputs is already a list of inputs
                    if all(isinstance(m, list) for m in inputs):
                        # Check if this is a batch (list of list of messages)
                        if any(isinstance(x, list) for x in inputs):
                            selected_batch = [(uuid, entry)]
                            # Put back all others
                            for _uuid, _entry in raw_requests:
                                if _uuid != uuid:
                                    retry_count = _entry.get("retry_count", 0)
                                    max_retries = _entry.get("max_retries", 3)
                                    self.enqueue_request(type, model_name, _uuid, _entry["payload"], retry_count=retry_count, max_retries=max_retries)
                            break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset((k, v) for k, v in payload.items() if k != "inputs")
                    grouped[key].append((uuid, entry))
                else:
                    # If not a pre-batched request, pick one uniform group (up to 2)
                    selected_batch = []
                    for batch in grouped.values():
                        selected_batch = batch[:8]
                        break

                    # Re-enqueue unused or extra requests
                    for batch in grouped.values():
                        if batch != selected_batch:
                            for uuid, entry in batch:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                        elif len(batch) > 8:
                            for uuid, entry in batch[8:]:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)

                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[List[Dict[str, str]]] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            inputs: List[Dict[str, str]] | List[List[Dict[str, str]]] = entry["payload"]["inputs"]
                            if all(isinstance(x, list) for x in inputs):
                                input_texts.extend(inputs)
                                response_counts.append((uuid, entry, len(inputs)))
                            else:
                                input_texts.append(inputs)
                                response_counts.append((uuid, entry, 1))

                        # Get generation parameters
                        config = {k:v for k, v in selected_batch[0][1]["payload"].items() if k != "inputs"}

                        with torch.no_grad():
                            responses = pipe(input_texts, **config)
                        torch.cuda.empty_cache()

                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index: index + count]
                            generated_texts: List[str] = [o[0]["generated_text"][-1]["content"] for o in outputs]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: str|List[str] = generated_texts if count > 1 or isinstance(entry["payload"]["inputs"][0], list) else generated_texts[0]

                            self.add_to_completed(type, model_name, uuid, entry, response=final_response)
                    except Exception as e:
                        logger.error("Error occurred in %s: %s", model_name, e)
                        for uuid, entry in selected_batch:
                            retry_count = entry.get("retry_count", 0) + 1
                            self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    pipe = None
                    break
    
    def vllmTextGenerationWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting vllm text generation worker for {type}/{model_name}")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()
        pipe = None
        while True:
            now = time.time()
            if not pipe:
                if self.ready_requests[type][model_name]:
                    logger.info(f"Attempting to load model {model_name}")
                    try:
                        with self._gpuLock, self._gpuLockM:
                            available_gpus = get_optimal_gpu_set(model_name, dtype="fp16")
                            logger.debug(f"Available GPUs for {model_name}: {available_gpus}")
                            logger.info(f"Loading model {model_name} on GPU: {available_gpus[0]}")
                            start_load_time = time.time()
                            
                            pipe = LLM(
                                model=model_name,
                                tensor_parallel_size=2,  # Use both GPUs
                                max_model_len=32768,
                                gpu_memory_utilization=0.8,
                                # swap_space=16,
                                # enable_chunked_prefill=True,
                                max_num_batched_tokens=256000,
                                # max_num_seqs=512,
                                # max_paddings=512,
                                # block_size=32,
                                # enable_prefix_caching=True,
                                enable_chunked_prefill=True,
                                # max_num_batched_tokens_in_prefill=8192,
                                enforce_eager=False,
                                # disable_custom_all_reduce=False,
                                # trust_remote_code=True,
                                # use_v2_block_manager=True,
                                # enable_lora=False,
                                # scheduler_policy="fcfs",
                                # preemption_mode="recompute",
                            )
                            
                            load_time = time.time() - start_load_time
                            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                            # else:
                            #     logger.warning(f"No available GPUs for {model_name}, waiting...")
                            #     time.sleep(60)  # Wait before retry
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {str(e)}")
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        time.sleep(60)  # Wait before retry
                        continue
                                
            if pipe and self.ready_requests[type][model_name]:
                raw_requests = self.dequeue_request(type, model_name, 32)
                grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
                
                for uuid, entry in raw_requests:
                    payload = entry["payload"]
                    sentences: List[str|Dict[str, Any]]|str|Dict[str, Any] = payload["inputs"]
                    
                     # Handle pre-batched requests: messages is already a list of messages
                    if isinstance(sentences, list):
                        selected_batch = [(uuid, entry)]
                        # Put back all others
                        for _uuid, _entry in raw_requests:
                            if _uuid != uuid:
                                retry_count = _entry.get("retry_count", 0)
                                self.enqueue_request(type, model_name, _uuid, _entry["payload"], retry_count=retry_count)
                        break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset((k, v) for k, v in payload.items() if k != "inputs")
                    grouped[key].append((uuid, entry))
                else:
                    # If not a pre-batched request, pick one uniform group (up to 8)
                    selected_batch = []
                    for batch in grouped.values():
                        selected_batch = batch[:16]
                        break

                    # Re-enqueue unused or extra requests
                    for batch in grouped.values():
                        if batch != selected_batch:
                            for uuid, entry in batch:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                        elif len(batch) > 16:
                            for uuid, entry in batch[16:]:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                
                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[str|Dict[str, Any]] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            inputs: List[str|Dict[str, Any]]|str|Dict[str, Any] = entry["payload"]["inputs"]
                            if isinstance(inputs, list):
                                input_texts.extend(inputs)
                                response_counts.append((uuid, entry, len(inputs)))
                            else:
                                input_texts.append(inputs)
                                response_counts.append((uuid, entry, 1))
                        
                        # Get generation parameters
                        config = {k:v for k, v in selected_batch[0][1]["payload"].items() if k != "inputs"}
                        sampling_params = SamplingParams(
                            **config,
                            # use_beam_search=config.get("use_beam_search", False),
                            # early_stopping=config.get("early_stopping", True),
                            skip_special_tokens=config.get("skip_special_tokens", True),
                            spaces_between_special_tokens=config.get("spaces_between_special_tokens", False),
                        )
                        with torch.no_grad():
                            for input_text in input_texts:
                                if isinstance(input_text, dict):
                                    if "multi_modal_data" in input_text:
                                        if "image" in input_text["multi_modal_data"]:
                                            logger.info(f'{input_text["multi_modal_data"]["image"]}')
                                            input_text["multi_modal_data"]["image"] = [to_pil_image(input_text["multi_modal_data"]["image"])]
                            responses = pipe.generate(input_texts, sampling_params)
                        torch.cuda.empty_cache()
                        
                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index: index + count]
                            generated_texts: List[str] = [o.outputs[0].text for o in outputs]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: str|List[str] = generated_texts if count > 1 or isinstance(entry["payload"]["inputs"], list) else generated_texts[0]

                            self.add_to_completed(type, model_name, uuid, entry, response=final_response)
                    except Exception as e:
                        logger.error("Error occurred in %s: %s", model_name, e)
                        for uuid, entry in selected_batch:
                            retry_count = entry.get("retry_count", 0) + 1
                            self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                        time.sleep(60)
            else:
                if time.time() - last_active > 600 and not self.ready_requests[type][model_name]:
                    pipe = None
                    torch.distributed.destroy_process_group()
                    break
        
    def sentTransWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        last_active = time.time()
        pipe = None
        while True:
            now = time.time()
            if not pipe:
                if self.ready_requests[type][model_name]:
                    with self._gpuLock, self._gpuLockM:
                        available_gpus = get_optimal_gpu_set(model_name, dtype="fp16")
                        if available_gpus:
                            pipe = SentenceTransformer(model_name, device=f"cuda:{available_gpus[0]}", trust_remote_code=True)
                            pipe.eval()
            if pipe and self.ready_requests[type][model_name]:
                raw_requests = self.dequeue_request(type, model_name, 64)
                grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
                
                for uuid, entry in raw_requests:
                    payload = entry["payload"]
                    sentences: List[str]|str = payload["sentences"]

                     # Handle pre-batched requests: messages is already a list of messages
                    if isinstance(sentences, list):
                        selected_batch = [(uuid, entry)]
                        # Put back all others
                        for _uuid, _entry in raw_requests:
                            if _uuid != uuid:
                                retry_count = _entry.get("retry_count", 0)
                                self.enqueue_request(type, model_name, _uuid, _entry["payload"], retry_count=retry_count)
                        break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset((k, v) for k, v in payload.items() if k != "sentences")
                    grouped[key].append((uuid, entry))
                else:
                    # If not a pre-batched request, pick one uniform group (up to 8)
                    selected_batch = []
                    for batch in grouped.values():
                        selected_batch = batch[:32]
                        break

                    # Re-enqueue unused or extra requests
                    for batch in grouped.values():
                        if batch != selected_batch:
                            for uuid, entry in batch:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                        elif len(batch) > 32:
                            for uuid, entry in batch[32:]:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                
                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[str] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            sentences: str | List[str] = entry["payload"]["sentences"]
                            if isinstance(sentences, list):
                                input_texts.extend(sentences)
                                response_counts.append((uuid, entry, len(sentences)))
                            else:
                                input_texts.append(sentences)
                                response_counts.append((uuid, entry, 1))
                        
                        # Get generation parameters
                        config = {k:v for k, v in selected_batch[0][1]["payload"].items() if k != "sentences"}
                        
                        with torch.no_grad():
                            responses = pipe.encode(input_texts, **config)
                        torch.cuda.empty_cache()
                        
                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index: index + count]
                            generated_texts: List[List[float]] = [o.tolist() for o in outputs]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: List[float]|List[List[float]] = generated_texts if count > 1 or isinstance(entry["payload"]["sentences"], list) else generated_texts[0]

                            self.add_to_completed(type, model_name, uuid, entry, response=final_response)
                    except Exception as e:
                        logger.error("Error occurred in %s: %s", model_name, e)
                        for uuid, entry in selected_batch:
                            retry_count = entry.get("retry_count", 0) + 1
                            self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    pipe = None
                    break

    def CLIPWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        last_active = time.time()
        model = None
        processor = None
        device = None
        while True:
            now = time.time()
            if not model:
                if self.ready_requests[type][model_name]:
                    with self._gpuLock, self._gpuLockM:
                        available_gpus = get_optimal_gpu_set(model_name, dtype="fp16")
                        if available_gpus:
                            model = CLIPModel.from_pretrained(model_name)
                            processor = CLIPProcessor.from_pretrained(model_name)
                            device = "cuda:{}".format(available_gpus[0])
                            model.to(device)
            
            if model and processor and device and self.ready_requests[type][model_name]:
                raw_requests = self.dequeue_request(type, model_name)[0]
                if raw_requests:
                    last_active = now
                    uuid, entry = raw_requests
                    try:                        
                        features = entry["payload"]["features"]
                        if features == "image":
                            logger.debug("Processing input images")
                            inputs = model(images=entry["payload"]["images"], return_tensors="pt").to(device)
                            with torch.no_grad():
                                logger.debug("Generating image features")
                                image_features = model.get_image_features(**inputs)
                            response = image_features.cpu().numpy().astype('float32').tolist()
                        elif features == "text":
                            logger.debug("Processing input texts")
                            inputs = model(text=entry["payload"]["text"], return_tensors="pt").to(device)
                            with torch.no_grad():
                                logger.debug("Generating text features")
                                text_features = model.get_text_features(**inputs)
                            response = text_features.cpu().numpy().astype('float32').tolist()
                        else:
                            logger.error("Invalid features: %s", features)
                            self.add_to_completed(type, model_name, uuid, entry, error="Invalid features: %s" % features)
                            continue              
                        torch.cuda.empty_cache()
                        
                        self.add_to_completed(type, model_name, uuid, entry, response=response)
                    except Exception as e:
                        logger.info("Error occured in %s: %s", model_name, e)
                        retry_count = entry.get("retry_count", 0) + 1
                        self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    model, processor, device = None, None, None
                    break
                
    def imageTextToTextWorker(self, **kwargs: Any):
        type: str = kwargs.get("type", "")
        model_name: str = kwargs.get("model_name", "")
        logger.info(f"Starting text generation worker for {type}/{model_name}")
        logger.debug(f"Worker config: {kwargs}")
        last_active = time.time()
        pipe = None
        while True:
            now = time.time()
            if not pipe:
                if self.ready_requests[type][model_name]:
                    logger.info(f"Attempting to load model {model_name}")
                    try:
                        with self._gpuLock, self._gpuLockM:
                            available_gpus = get_optimal_gpu_set(model_name, dtype="fp16")
                            logger.debug(f"Available GPUs for {model_name}: {available_gpus}")
                            if available_gpus:
                                logger.info(f"Loading model {model_name} on GPU: {available_gpus[0]}")
                                start_load_time = time.time()
                                pipe = pipeline(
                                    "image-text-to-text",
                                    model=model_name,
                                    device_map=f"cuda:{available_gpus[0]}",
                                    torch_dtype=torch.bfloat16,
                                    clean_up_tokenization_spaces=False,
                                    trust_remote_code=True,
                                    use_fast=True
                                )

                                load_time = time.time() - start_load_time
                                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
                            else:
                                logger.warning(f"No available GPUs for {model_name}, waiting...")
                                time.sleep(60)  # Wait before retry
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {str(e)}")
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        time.sleep(60)  # Wait before retry
                        continue
            if pipe and self.ready_requests[type][model_name]:
                logger.debug(f"Dequeuing up to 32 requests for {type}/{model_name}")
                raw_requests = self.dequeue_request(type, model_name, 32)
                grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

                for uuid, entry in raw_requests:
                    payload: Dict[str, Any] = entry["payload"]
                    inputs: List[Dict[str, str]] | List[List[Dict[str, str]]] = payload["inputs"]

                    # Handle pre-batched requests: inputs is already a list of inputs
                    if all(isinstance(m, list) for m in inputs):
                        # Check if this is a batch (list of list of messages)
                        if any(isinstance(x, list) for x in inputs):
                            selected_batch = [(uuid, entry)]
                            # Put back all others
                            for _uuid, _entry in raw_requests:
                                if _uuid != uuid:
                                    retry_count = _entry.get("retry_count", 0)
                                    max_retries = _entry.get("max_retries", 3)
                                    self.enqueue_request(type, model_name, _uuid, _entry["payload"], retry_count=retry_count, max_retries=max_retries)
                            break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset((k, v) for k, v in payload.items() if k != "inputs")
                    grouped[key].append((uuid, entry))
                else:
                    # If not a pre-batched request, pick one uniform group (up to 8)
                    selected_batch = []
                    for batch in grouped.values():
                        selected_batch = batch[:8]
                        break

                    # Re-enqueue unused or extra requests
                    for batch in grouped.values():
                        if batch != selected_batch:
                            for uuid, entry in batch:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)
                        elif len(batch) > 8:
                            for uuid, entry in batch[8:]:
                                retry_count = entry.get("retry_count", 0)
                                max_retries = entry.get("max_retries", 3)
                                self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count, max_retries=max_retries)

                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[List[Dict[str, str]]] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            inputs: List[Dict[str, str]] | List[List[Dict[str, str]]] = entry["payload"]["inputs"]
                            if all(isinstance(x, list) for x in inputs):
                                input_texts.extend(inputs)
                                response_counts.append((uuid, entry, len(inputs)))
                            else:
                                input_texts.append(inputs)
                                response_counts.append((uuid, entry, 1))

                        # Get generation parameters
                        config = {k:v for k, v in selected_batch[0][1]["payload"].items() if k != "inputs"}

                        with torch.no_grad():
                            responses = pipe(input_texts, **config)
                        torch.cuda.empty_cache()

                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index: index + count]
                            generated_texts: List[str] = [o[0]["generated_text"][-1]["content"] for o in outputs]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: str|List[str] = generated_texts if count > 1 or isinstance(entry["payload"]["inputs"][0], list) else generated_texts[0]

                            self.add_to_completed(type, model_name, uuid, entry, response=final_response)
                    except Exception as e:
                        logger.error("Error occurred in %s: %s", model_name, e)
                        for uuid, entry in selected_batch:
                            retry_count = entry.get("retry_count", 0) + 1
                            self.enqueue_request(type, model_name, uuid, entry["payload"], retry_count=retry_count)
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests[type][model_name]:
                    pipe = None
                    break