import os
import time
import json
import pickle
import tempfile
from multiprocessing import Process, Event, Queue
from typing import Optional, Any, Dict, List, Tuple

class IPCManger:
    """
    Manages inter-process communication between the main server and worker processes.
    Uses file-based communication to avoid pickling issues.
    """
    def __init__(self, base_dir: Optional[str]=None):
        if base_dir is None:
            self.base_dir = os.path.join(tempfile.gettempdir(), "model_server_ipc")
        else:
            self.base_dir = base_dir
            
        os.makedirs(self.base_dir, exist_ok=True)
        self.request_dir = os.path.join(self.base_dir, "requests")
        self.response_dir = os.path.join(self.base_dir, "responses")
        os.makedirs(self.request_dir, exist_ok=True)
        os.makedirs(self.response_dir, exist_ok=True)
        
    def send_request(self, worker_type: str, model_name: str, uuid: str, payload: Dict[str, Any]) -> None:
        """Send a request to a specific worker"""
        worker_dir = os.path.join(self.request_dir, f"{worker_type}_{model_name}")
        os.makedirs(worker_dir, exist_ok=True)
        
        request_file = os.path.join(worker_dir, f"{uuid}.json")
        with open(request_file, 'w') as f:
            json.dump({
                "uuid": uuid,
                "payload": payload,
                "timestamp": time.time()
            }, f)
            
    def get_pending_requests(self, worker_type: str, model_name: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all pending requests for a specific worker"""
        worker_dir = os.path.join(self.request_dir, f"{worker_type}_{model_name}")
        if not os.path.exists(worker_dir):
            return []
            
        requests = []
        for filename in os.listdir(worker_dir):
            if filename.endswith('.json'):
                uuid = filename[:-5]  # Remove .json extension
                file_path = os.path.join(worker_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        request_data = json.load(f)
                    requests.append((uuid, request_data))
                    # Remove the file after reading
                    os.remove(file_path)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Handle corrupted or already removed files
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
    def send_response(self, worker_type: str, model_name: str, uuid: str, response: Any) -> None:
        """Send a response for a specific request"""
        worker_dir = os.path.join(self.response_dir, f"{worker_type}_{model_name}")
        os.makedirs(worker_dir, exist_ok=True)
        
        response_file = os.path.join(worker_dir, f"{uuid}.json")
        with open(response_file, 'w') as f:
            # For complex objects that might not be JSON serializable, use pickle if needed
            try:
                json.dump({
                    "uuid": uuid,
                    "response": response,
                    "timestamp": time.time()
                }, f)
            except TypeError:
                # For non-JSON serializable objects, store a placeholder and use pickle in a separate file
                json.dump({
                    "uuid": uuid,
                    "response": "__PICKLED__",
                    "timestamp": time.time()
                }, f)
                with open(response_file + '.pickle', 'wb') as pf:
                    pickle.dump(response, pf)
    
    def get_response(self, worker_type: str, model_name: str, uuid: str, timeout: float = 60.0) -> Optional[Any]:
        """Get a response for a specific request with timeout"""
        worker_dir = os.path.join(self.response_dir, f"{worker_type}_{model_name}")
        if not os.path.exists(worker_dir):
            return None
            
        response_file = os.path.join(worker_dir, f"{uuid}.json")
        pickle_file = response_file + '.pickle'
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                try:
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                    
                    # Check if it's a pickled response
                    if response_data.get("response") == "__PICKLED__" and os.path.exists(pickle_file):
                        with open(pickle_file, 'rb') as pf:
                            response_data["response"] = pickle.load(pf)
                        os.remove(pickle_file)
                    
                    # Clean up response file
                    os.remove(response_file)
                    return response_data["response"]
                except (json.JSONDecodeError, FileNotFoundError, pickle.PickleError):
                    # Handle corrupted files
                    if os.path.exists(response_file):
                        os.remove(response_file)
                    if os.path.exists(pickle_file):
                        os.remove(pickle_file)
                    return None
            time.sleep(0.1)
        
        return None
    
# Example of how to update the Flask route in ModelServer to use IPC
def infer_with_ipc(ipc_manager: IPCManger, type: str, model_name: str, uuid: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Example of how to implement the infer route with IPC"""
    # Send request to worker via IPC
    ipc_manager.send_request(type, model_name, uuid, payload)
    
    # Wait for response with timeout
    response = ipc_manager.get_response(type, model_name, uuid, timeout=60.0)
    
    if response is None:
        return {"error": "Request timed out", "status": "FAILED"}, 504
    if isinstance(response, dict) and "error" in response:
        return response, 400
    return {"response": response, "status": "SUCCESS"}, 200