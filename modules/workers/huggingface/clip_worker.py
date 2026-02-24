from typing import Any
import time
from transformers import CLIPModel, CLIPProcessor
import torch

from . import logger
from ...base import Worker
from ...utils import get_optimal_gpu_set


class CLIPWorker(Worker):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.configure(**kwargs)

    def configure(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def worker(self, **kwargs: Any):
        last_active = time.time()
        model = None
        processor = None
        device = None
        while True:
            now = time.time()
            if not model:
                if self.ready_requests:
                    with self.shared["_gpuLockM"]:
                        available_gpus = get_optimal_gpu_set(
                            self.model_name, dtype="fp16"
                        )
                        if available_gpus:
                            model = CLIPModel.from_pretrained(self.model_name)
                            processor = CLIPProcessor.from_pretrained(self.model_name)
                            device = "cuda:{}".format(available_gpus[0])
                            model.to(device)

            if model and processor and device:
                raw_requests = self.dequeue_request()[0]
                if raw_requests:
                    last_active = now
                    uuid, entry = raw_requests
                    try:
                        features = entry["payload"]["features"]
                        if features == "image":
                            logger.debug("Processing input images")
                            inputs = model(
                                images=entry["payload"]["images"], return_tensors="pt"
                            ).to(device)
                            with torch.no_grad():
                                logger.debug("Generating image features")
                                image_features = model.get_image_features(**inputs)
                            response = (
                                image_features.cpu().numpy().astype("float32").tolist()
                            )
                        elif features == "text":
                            logger.debug("Processing input texts")
                            inputs = model(
                                text=entry["payload"]["text"], return_tensors="pt"
                            ).to(device)
                            with torch.no_grad():
                                logger.debug("Generating text features")
                                text_features = model.get_text_features(**inputs)
                            response = (
                                text_features.cpu().numpy().astype("float32").tolist()
                            )
                        else:
                            logger.error("Invalid features: %s", features)
                            with self._completedLock, self._completedLockM:
                                self.completed_requests[uuid] = {
                                    "error": "Invalid features: %s" % features,
                                    "timestamp": entry["timestamp"],
                                    "payload": entry["payload"],
                                }
                                self.completed_request_event.set()
                                self.completed_request_event.clear()
                            continue
                        torch.cuda.empty_cache()

                        self.add_to_completed(uuid, entry, response=response)
                        # with self._completedLock, self._completedLockM:
                        #     self.completed_requests[uuid] = {
                        #         "response": response,
                        #         "timestamp": entry["timestamp"],
                        #         "payload": entry["payload"]
                        #     }
                        #     self.completed_request_event.set()
                        #     self.completed_request_event.clear()
                    except Exception as e:
                        logger.info("Error occured in %s: %s", self.model_name, e)
                        self.enqueue_request(uuid, entry["payload"])
                        time.sleep(60)
            else:
                if time.time() - last_active > 300 and not self.ready_requests:
                    model, processor, device = None, None, None
                    break
