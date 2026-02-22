from typing import Any, Dict, List
import requests
import time
from dotenv import load_dotenv
import os
from multiprocessing import Process

from . import logger
from ..base import Worker

load_dotenv()

PORTS = [int(port) for port in os.environ.get("OLLAMA_PORTS", "").split(" ")]

def call_ollama(payload: Dict[str, Any], port: int):
    if port in [6001, 6002]:
        url = f"http://localhost:{port}/api/chat"
    else:
        raise ValueError(f"Invalid port: {port}")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

class OllamaWorker(Worker):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        self.configure(**kwargs)
        
    def configure(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def start_processor(self, **kwargs: Any) -> None:
        processes: List[Process] = []
        for port in PORTS:
            process = Process(target=super().start_processor, args=(port,))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        
    def worker(self, **kwargs: Any):
        port: int = kwargs.get("port", "")
        if not port:
            return
        last_active = time.time()
        while True:
            now = time.time()
            requests = self.dequeue_request()
            if requests:
                uuid, entry = requests[0]
                last_active = now
                try:
                    response = call_ollama(entry["payload"], port)
                    self.add_to_completed(uuid, entry, response=response)
                    # with self._completedLock, self._completedLockM:
                    #     if uuid in self.completed_requests:
                    #         self.completed_requests[uuid]["timestamp"] = entry["timestamp"]
                    #     else:
                    #         self.completed_requests[uuid] = {
                    #             "response": response,
                    #             "timestamp": entry["timestamp"],
                    #             "payload": entry["payload"]
                    #         }
                    #     self.completed_request_event.set()
                    #     self.completed_request_event.clear()
                except Exception as e:
                    logger.error("Error occurred in %s: %s", port, e)
                    self.enqueue_request(uuid, entry["payload"])
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests:
                    break