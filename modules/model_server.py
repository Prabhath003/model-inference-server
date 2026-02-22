import os
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Manager, Process, Event
from multiprocessing.managers import DictProxy, ListProxy
import threading
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional, Tuple
import hashlib
import json
import psutil
import time
import threading

from .workers import CLIPWorker, ImageTextToTextWorker, SentTransWorker, TextGenerationBnbWorker, TextGenerationWorker, GenAIWorker, OllamaWorker, OpenAIWorker
from .base import Worker
from .utils import get_available_gpus
from .ipc import IPCManger, infer_with_ipc

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

class ModelServer:
    def __init__(self):
        self.manager = Manager()
        self.ipc_manager = IPCManger()
        
        # For managing worker creation requests
        self._workerLock = self.manager.Lock()
        self._requestLock = self.manager.Lock()
        self.workers_config = self.manager.dict()  # Just stores configuration, not actual workers
        self.workers_procs = self.manager.dict()
        self.request_queue = self.manager.list()
        
    def printStats(self):
        while True:
            time.sleep(60)
            logger.info(f"Memory usage: {psutil.virtual_memory().percent:.2f}%")
            logger.info(f"GPU usage: {get_available_gpus()}")
            logger.info(f"CPU usage: {psutil.cpu_percent(interval=1):.2f}%")
        
    def create_worker(self):
        while True:
            request = self.dequeue_request()
            if request:
                type, model_name = request
                with self._workerLock:
                    if type not in self.workers_config:
                        self.workers_config[type] = self.manager.dict()
                        self.workers_procs[type] = self.manager.dict()
                    if model_name not in self.workers_config[type]:
                        # Store config for this worker
                        worker_config: Dict[str, Any] = {
                            "model_name": model_name,
                            "type": type,
                            "max_workers": 4 if model_name == "Qwen/Qwen2.5-14B-Instruct" else 1
                        }
                        self.workers_config[type][model_name] = worker_config
                        
                        # Map worker types to their process creation functions
                        worker_process_map = {
                            "text-generation": self._run_text_generation_worker,
                            "image-text-to-text": self._run_image_text_to_text_worker,
                            "sent-trans": self._run_sent_trans_worker,
                            "text-generation-bnb": self._run_text_generation_bnb_worker,
                            "clip": self._run_clip_worker,
                            "genai": self._run_genai_worker,
                            "ollama": self._run_ollama_worker,
                            "openai": self._run_openai_worker
                        }
                        
                        # Create a separate process for each worker type
                        if type == "text-generation":
                            p = Process(target=self._run_text_generation_worker, args=(model_name, worker_config))
                            logger.info(f"Created text-generation worker for {model_name}")
                        elif type == "image-text-to-text":
                            p = Process(target=self._run_image_text_to_text_worker, args=(model_name, worker_config))
                            logger.info(f"Created image-text-to-text worker for {model_name}")
                        elif type == "sent-trans":
                            p = Process(target=self._run_sent_trans_worker, args=(model_name, worker_config))
                            logger.info(f"Created sent-trans worker for {model_name}")
                        elif type == "text-generation-bnb":
                            p = Process(target=self._run_text_generation_bnb_worker, args=(model_name, worker_config))
                            logger.info(f"Created text-generation-bnb worker for {model_name}")
                        elif type == "clip":
                            p = Process(target=self._run_clip_worker, args=(model_name, worker_config))
                            logger.info(f"Created CLIP worker for {model_name}")
                        elif type == "genai":
                            p = Process(target=self._run_genai_worker, args=(model_name, worker_config))
                            logger.info(f"Created genai worker for {model_name}")
                        elif type == "ollama":
                            p = Process(target=self._run_ollama_worker, args=(model_name, worker_config))
                            logger.info(f"Created ollama worker for {model_name}")
                        elif type == "openai":
                            p = Process(target=self._run_openai_worker, args=(model_name, worker_config))
                            logger.info(f"Created openai worker for {model_name}")
                        else:
                            logger.error(f"Unsupported model type: {type}")
                        
                        self.workers_procs[type][model_name] = p
                        p.start()
            else:
                self.new_request_event.wait()
                self.new_request_event.clear()
                
    # Individual worker runner methods that create the worker inside the process
    def _run_text_generation_worker(self, model_name: str, config: Dict[str, Any]):
        worker = TextGenerationWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_image_text_to_text_worker(self, model_name: str, config: Dict[str, Any]):
        worker = ImageTextToTextWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_sent_trans_worker(self, model_name: str, config: Dict[str, Any]):
        worker = SentTransWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_text_generation_bnb_worker(self, model_name: str, config: Dict[str, Any]):
        worker = TextGenerationBnbWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_clip_worker(self, model_name: str, config: Dict[str, Any]):
        woraker = CLIPWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_genai_worker(self, model_name: str, config: Dict[str, Any]):
        worker = GenAIWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_ollama_worker(self, model_name: str, config: Dict[str, Any]):
        worker = OllamaWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
        
    def _run_openai_worker(self, model_name: str, config: Dict[str, Any]):
        worker = OpenAIWorker(model_name=model_name, max_workers=config["max_workers"], type=config["type"])
        worker.start_processor()
                
    def enqueue_request(self, type: str, model_name: str):
        with self._requestLock:
            if not self.check_worker_created(type, model_name):
                self.request_queue.append((type, model_name))
                self.new_request_event.set()
            
    def dequeue_request(self) -> Optional[Tuple[str, str]]:
        with self._requestLock:
            if self.request_queue:
                return self.request_queue.pop(0)
        return None
            
    def check_worker_created(self, type: str, model_name: str) -> bool:
            return type in self.workers and model_name in self.workers[type]
        
    def start_flask_server(self):
        app = Flask(__name__)
        app.secret_key = "model-server"
        app.config["SESSION_TYPE"] = "filesystem"
        
        @app.route("/infer", methods=["POST"])
        def infer():
            data: Optional[Dict[str, Any]] = request.json
            if not data:
                return jsonify({"error": "Missing payload"}), 400
            payload = data.get("payload")
            model_name = data.get("model_name")
            type = data.get("type")
            if not payload or not model_name or not type:
                logger.error("Missing payload, model_name, or type in request")
                return jsonify({"error": "Missing payload, model_name, or type"}), 400
            
            self.enqueue_request(type, model_name)
            while not self.check_worker_created(type, model_name):
                continue
            
            # We need to implement communication with workers via some shared state or IPC
            # This part needs to be redesigned to properly communicate with worker processes
            # For now, we'll return a placeholder error
            return jsonify({"error": "Worker communication not implemented yet", "status": "FAILED"}), 501
            
        app.run(host="0.0.0.0", port=1121)
        
    def start(self):
        stats_thread = threading.Thread(target=self.printStats, daemon=True)
        stats_thread.start()
        worker_creater = Process(target=self.create_worker)
        worker_creater.start()
        # flask_server = Process(target=self.start_flask_server)
        # flask_server.start()
        self.start_flask_server()
            
