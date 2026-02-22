from typing import Any, List, Dict, Tuple
import time
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict

from . import logger
from ...base import Worker
from ...utils import get_optimal_gpu_set

class SentTransWorker(Worker):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        self.configure(**kwargs)

    def configure(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def worker(self, **kwargs: Any):
        last_active = time.time()
        pipe = None
        while True:
            now = time.time()
            if not pipe:
                if self.ready_requests:
                    with self.shared["_gpuLockM"]:
                        available_gpus = get_optimal_gpu_set(self.model_name, dtype="fp16")
                        if available_gpus:
                            pipe = SentenceTransformer(self.model_name, device=f"cuda:{available_gpus[0]}", trust_remote_code=True)
                            pipe.eval()
            if pipe:
                raw_requests = self.dequeue_request(64)
                grouped: Dict[frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
                
                for uuid, entry in raw_requests:
                    payload = entry["payload"]
                    messages = payload["messages"]
                    
                     # Handle pre-batched requests: messages is already a list of messages
                    if isinstance(messages, list) and all(isinstance(m, str) for m in messages):
                        # Check if this is a batch (list of list of messages)
                        if any(isinstance(x, list) for x in messages):
                            selected_batch = [(uuid, entry)]
                            # Put back all others
                            for _uuid, _entry in raw_requests:
                                if _uuid != uuid:
                                    self.enqueue_request(_uuid, _entry["payload"])
                            break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset((k, v) for k, v in payload.items() if k != "messages")
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
                                self.enqueue_request(uuid, entry["payload"])
                        elif len(batch) > 32:
                            for uuid, entry in batch[32:]:
                                self.enqueue_request(uuid, entry["payload"])
                
                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[List[str]] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = []  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            messages: List[str] | List[List[str]] = entry["payload"]["messages"]
                            if all(isinstance(x, list) for x in messages):
                                input_texts.extend(messages)
                                response_counts.append((uuid, entry, len(messages)))
                            else:
                                input_texts.append(messages)
                                response_counts.append((uuid, entry, 1))
                        
                        with torch.no_grad():
                            responses = pipe.encode(input_texts)
                        torch.cuda.empty_cache()
                        
                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index: index + count]
                            generated_texts: List[List[float]] = [o.tolist() for o in outputs]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: List[float]|List[List[float]] = generated_texts if count > 1 else generated_texts[0]

                            self.add_to_completed(uuid, entry, response=final_response)
                            # with self._completedLock, self._completedLockM:
                            #     self.completed_requests[uuid] = {
                            #         "response": final_response,
                            #         "timestamp": entry["timestamp"],
                            #         "payload": entry["payload"]
                            #     }
                            #     self.completed_request_event.set()
                            #     self.completed_request_event.clear()
                    except Exception as e:
                        logger.error("Error occurred in %s: %s", self.model_name, e)
                        for uuid, entry in selected_batch:
                            self.enqueue_request(uuid, entry["payload"])
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests:
                    pipe = None
                    break