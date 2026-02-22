from typing import Any, Dict, List
import requests
import time
from dotenv import load_dotenv
import os
from multiprocessing import Process

from . import logger
from ..base import Worker
from ..twilio_communication import send_message

load_dotenv()

API_KEYS = os.environ.get("OPENAI_API_KEYS", "").split(" ")

def call_openai(payload: Dict[str, Any], api_key: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

class OpenAIWorker(Worker):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        self.configure(**kwargs)
        
    def configure(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def start_processor(self, **kwargs: Any) -> None:
        processes: List[Process] = []
        for api_key in API_KEYS:
            process = Process(target=super().start_processor, args=(api_key, {
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
        
    def worker(self, **kwargs: Any):
        api_key: str = kwargs.get("api_key", "")
        shared_data: Dict[str, Any] = kwargs.get("shared_data", {})
        if not api_key or not shared_data:
            return
        last_active = time.time()
        while True:
            now = time.time()
            with shared_data['counts_lock']:
                if now - shared_data['start_time'].value > 86400:
                    send_message("openai", "daily_limit resetted")
                    shared_data['total_requests_count'].value = 0
                    shared_data['start_time'].value = now
                if now - shared_data['minute_start_time'].value > 60:
                    shared_data['minute_requests_count'].value = 0
                    shared_data['minute_start_time'].value = now

                if (shared_data['minute_requests_count'].value < 500 and
                        shared_data['total_requests_count'].value < 10000):
                    requests = self.dequeue_request()
                    if requests:
                        uuid, entry = requests[0]
                        shared_data['minute_requests_count'].value += 1
                        shared_data['total_requests_count'].value += 1
                    else:
                        uuid, entry = None, None
                else:
                    if shared_data['total_requests_count'].value >= 10000:
                        send_message("openai", "crossed_daily_limit")
                    uuid = None
                    entry = None

            if uuid and entry:
                last_active = now
                try:
                    response = call_openai(entry["payload"], api_key)
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
                    logger.error("Error occurred in %s: %s", api_key, e)
                    self.enqueue_request(uuid, entry["payload"])
                    if "429" in str(e):
                        time.sleep(600)
                    time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests:
                    break