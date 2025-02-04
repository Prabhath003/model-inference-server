import logging
import os
import time
import threading
import requests
from typing import Literal
from multiprocessing import Process, Manager
import random
from typing import Any
from dotenv import load_dotenv

from .model_server import ModelServer
from .utils import get_available_ports

load_dotenv()

manager = Manager()
processes = manager.dict()
lock = threading.Lock()

def monitor_processes():
    while True:
        current_time = time.time()
        with lock:
            for key, value in list(processes.items()):
                if current_time - value["last_response_time"] > 600:  # 10 minutes
                    logging.warning(f"Killing process for {key} due to inactivity.")
                    os.kill(value["pid"], 9)  # Forcefully terminate the process
                    del processes[key]
        time.sleep(60)

monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
monitor_thread.start()

def inferModelInstance(data: dict[str, Any]):
    model_name: str = data["model_name"]
    model_type: str = data["type"]
    with lock:
        port = processes[f"{model_name}_{model_type}"]["port"]
        processes[f"{model_name}_{model_type}"]["last_response_time"] = time.time()
    url = f"http://localhost:{port}/infer"
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                logging.info(f"Inference request successful on port {port}")
                with lock:
                    processes[f"{model_name}_{model_type}"]["last_response_time"] = time.time()
                break
        except Exception as e:
            retries += 1
            logging.warning(f"{e} on {port}, retrying... ({retries}/{max_retries})")
            time.sleep(2)
    else:
        logging.error(f"Failed to get a successful, response after {max_retries} retries.")
        response = None
    return response

def wait_for_server_start(port: int):
    url = f"http://localhost:{port}/check_health"
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logging.info(f"Server started successfully on port {port}")
                break
        except requests.ConnectionError:
            logging.error(f"Connection error on port {port}, retrying...")
            time.sleep(5)
            
def createModelEndpoint(model_name: str, type: Literal["text-generation", "openai", "gen-ai", "sent-trans", "ollama"]="text-generation"):
    with lock:
        if f"{model_name}_{type}" in processes:
            logging.info(f"Model endpoint for {model_name} of type {type} already exists.")
            return
        logging.info(f"Creating model endpoint for model: {model_name}, type: {type}")
        available_ports = get_available_ports(int(os.environ.get("PORT_RANGE_START", 5001)), int(os.environ.get("PORT_RANGE_END", 5100)))
        if not available_ports:
            logging.error("No available ports to create model endpoint.")
            raise ConnectionRefusedError
        port = random.choice(available_ports)
        server = ModelServer(port, model_name, type)
        server_process = Process(target=server.start_server)
        server_process.start()
        processes[f"{model_name}_{type}"] = {"pid": server_process.pid, "last_response_time": time.time(), "port": port}

        wait_for_server_start(port)