from typing import Any, List, Dict, Tuple, Optional
from multiprocessing import Process, Manager
from threading import Thread
import time
import os 
import logging
from logging.handlers import RotatingFileHandler

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

class Worker:
    shared: Dict[str, Any]
    def __init__(self, model_name: str, max_workers: int, type: str):
        self.model_name = model_name
        self.max_workers = max_workers
        self.type = type
        
        # Create a manager for sharing state between processes
        self.manager = Manager()
        
        # Use multiprocessing locks instead of threading locks
        self._gpuLock = self.manager.Lock()
        self._completedLock = self.manager.Lock()
        self._requestLock = self.manager.Lock()
        
        # Use manager managed shared data structures
        self.ready_requests = self.manager.dict()
        self.completed_requests = self.manager.dict()
        
        # Events for signaling
        self.new_request_event = self.manager.Event()
        self.completed_request_event = self.manager.Event()
        
        # self.configure(**kwargs)
        
        self.start_time = time.time()
        
        stats_thread = Thread(target=self.printStats, daemon=True)
        stats_thread.start()
        # cleaner_thread = Thread(target=self.clean_complete_requests, daemon=True)
        # cleaner_thread.start()
        
    # def configure(self, **kwargs: Any):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)
            
    def printStats(self):
        while True:
            time.sleep(60)
            logger.info(f"Total requests: {len(self.ready_requests)} in {self.model_name} of {self.type}")
            logger.info(f"Completed requests: {len(self.completed_requests)} in {self.model_name} of {self.type}")
            logger.info(f"Time elapsed: {time.time() - self.start_time:.2f} seconds in {self.model_name} of {self.type}")
            logger.info(f"Average response time: {(sum(response['response_time'] for response in self.completed_requests.values()) / len(self.completed_requests)) if self.completed_requests else 0:.2f} seconds in {self.model_name} of {self.type}")
            
    # def clean_complete_requests(self):
    #     """
    #     Clean up completed queries that are older than one hour.
    #     """
    #     while True:
    #         time.sleep(3600)
    #         uuids = list(self.completed_requests.keys())
    #         for uuid in uuids:
    #             if time.time() - self.completed_requests[uuid]["timestamp"] > 3600:
    #                 with self._completedLock, self._completedLockM:
    #                     self.completed_requests.pop(uuid)
            
    def enqueue_request(self, uuid: str, payload: Dict[str, Any]) -> None:
        """Enqueue a request for processing"""
        with self._completedLock:
            if uuid in self.completed_requests:
                logger.info("Payload already processed for %s", uuid)
                return
        
        with self._requestLock:
            self.ready_requests[uuid] = {
                "payload": payload,
                "timestamp": time.time()
            }
            self.new_request_event.set()
            
    def dequeue_request(self, batch_size: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
        """Dequeue a batch of requests for processing"""
        with self._requestLock:
            if self.ready_requests:
                requests = list(self.ready_requests.keys())[:batch_size]
                return [(key, self.ready_requests.pop(key)) for key in requests]
        return []
    
    def get_response(self, uuid: str, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Get response for a request with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.completed_request_event.wait()
            with self._completedLock:
                if uuid in self.completed_requests:
                    response = self.completed_requests[uuid]["response"]
                    del self.completed_requests[uuid]
                    return response
        return None
    
    def add_to_completed(self, uuid: str, entry: Dict[str, Any], **kwargs: Any) -> None:
        with self._completedLock:
            self.completed_requests[uuid] = {
                "timestamp": entry["timestamp"],
                "response_time": time.time() - entry["timestamp"],
                "payload": entry["payload"],
                **kwargs
            }
            self.completed_request_event.set()
            self.completed_request_event.clear()

    def start_processor(self, **kwargs: Any) -> None:
        workers: List[Process] = []  
        while True:
            workers = [w for w in workers if w.is_alive()]
            queue_length = len(self.ready_requests)
            ideal_workers = min(self.max_workers, (queue_length + 9) // 10)
            
            if len(workers) < ideal_workers:
                for _ in range(ideal_workers - len(workers)):
                    p = Process(target=self.worker, args=(kwargs,), daemon=True)
                    p.start()
                    workers.append(p)
            self.new_request_event.wait()        
        
    def worker(self, **kwargs: Any) -> None:
        """Worker implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement worker method")
        