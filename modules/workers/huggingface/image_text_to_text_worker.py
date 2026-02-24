from typing import Any, List, Dict, Tuple
import time
from transformers import pipeline
import torch
from collections import defaultdict

from . import logger
from ...base import Worker
from ...utils import get_optimal_gpu_set


class ImageTextToTextWorker(Worker):
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
                        available_gpus = get_optimal_gpu_set(
                            self.model_name, dtype="fp16"
                        )
                        if available_gpus:
                            pipe = pipeline(
                                "image-text-to-text",
                                model=self.model_name,
                                device_map=f"cuda:{available_gpus[0]}",
                                torch_dtype=torch.bfloat16,
                                clean_up_tokenization_spaces=False,
                                trust_remote_code=True,
                                use_fast=True,
                            )
            if pipe:
                raw_requests = self.dequeue_request(32)
                grouped: Dict[
                    frozenset[tuple[Any, Any]], List[Tuple[str, Dict[str, Any]]]
                ] = defaultdict(list)

                for uuid, entry in raw_requests:
                    payload = entry["payload"]
                    messages = payload["messages"]

                    # Handle pre-batched requests: messages is already a list of messages
                    if isinstance(messages, list) and all(
                        isinstance(m, dict) for m in messages
                    ):
                        # Check if this is a batch (list of list of messages)
                        if any(isinstance(x, list) for x in messages):
                            selected_batch = [(uuid, entry)]
                            # Put back all others
                            for _uuid, _entry in raw_requests:
                                if _uuid != uuid:
                                    self.enqueue_request(_uuid, _entry["payload"])
                            break
                    # Exclude `messages` from the payload key for grouping
                    key = frozenset(
                        (k, v) for k, v in payload.items() if k != "messages"
                    )
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
                                self.enqueue_request(uuid, entry["payload"])
                        elif len(batch) > 8:
                            for uuid, entry in batch[8:]:
                                self.enqueue_request(uuid, entry["payload"])

                if selected_batch:
                    last_active = now
                    try:
                        # Prepare batched input
                        input_texts: List[List[Dict[str, str]]] = []
                        response_counts: List[Tuple[str, Dict[str, Any], int]] = (
                            []
                        )  # to track how many inputs per request
                        for uuid, entry in selected_batch:
                            messages: (
                                List[Dict[str, str]] | List[List[Dict[str, str]]]
                            ) = entry["payload"]["messages"]
                            if all(isinstance(x, list) for x in messages):
                                input_texts.extend(messages)
                                response_counts.append((uuid, entry, len(messages)))
                            else:
                                input_texts.append(messages)
                                response_counts.append((uuid, entry, 1))

                        # Get generation parameters
                        payload = selected_batch[0][1]["payload"]
                        temperature = payload.get("temperature", 0.1)
                        max_new_tokens = payload.get("max_new_tokens", 8192)

                        with torch.no_grad():
                            responses = pipe(
                                input_texts,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                            )
                        torch.cuda.empty_cache()

                        # Assign responses correctly
                        index = 0
                        for uuid, entry, count in response_counts:
                            outputs = responses[index : index + count]
                            generated_texts: List[str] = [
                                o[0]["generated_text"][-1]["content"] for o in outputs
                            ]
                            index += count

                            # If multiple outputs, store the list; otherwise a single response
                            final_response: str | List[str] = (
                                generated_texts if count > 1 else generated_texts[0]
                            )

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
