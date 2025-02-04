from typing import Literal
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI
import google.generativeai as genai
from flask import Flask, request, jsonify
from queue import Queue
from threading import Thread, Lock
import logging

from .utils import get_optimal_gpu_set

class ModelServer:
    def __init__(self, port: int, model_name: str, type: Literal["text-generation", "openai", "gen-ai", "sent-trans", "ollama"]="text-generation"):
        self.model_name = model_name
        self.type = type
        self.port = port
        logging.info(f"Initialized ModelServer with model: {model_name}, type: {type}, port: {port}")

    def start_server(self):
        app = Flask(__name__)
        if self.type in ["text-generation", "sent-trans"]:
            if "/" in self.model_name:
                available_gpus = get_optimal_gpu_set(self.model_name, dtype="fp16")
            else:
                available_gpus = get_optimal_gpu_set(
                    f"sentence-transformers/{self.model_name}", dtype="fp16"
                )
        if self.type == "text-generation":
            if not available_gpus:
                logging.error("No available GPUs to load the model.")
                raise RuntimeError("No available GPUs to load the model.")
            self.start_text_generation_server(app, available_gpus)
        elif self.type == "sent-trans":
            if not available_gpus:
                logging.error("No available GPUs to load the model.")
                raise RuntimeError("No available GPUs to load the model.")
            self.start_sent_trans_server(app, available_gpus)
        elif self.type == "openai":
            self.start_openai_server(app)
        elif self.type == "gen-ai":
            self.start_genai_server(app)
        elif self.type == "ollama":
            self.start_ollama_server(app)
        else:
            logging.error(f"Invalid model type: {self.type}")
            raise ValueError(f"Invalid model type: {self.type}")

        @app.route("/check_health", methods=["GET"])
        def check_health():
            return jsonify({"status": "healthy"}), 200
        
        logging.info(f"Starting server on port {self.port}")
        app.run(port=self.port)
        
    def start_text_generation_server(self, app: Flask, available_gpus: list[int]):
        try:
            model = pipeline(
                self.type,
                model=self.model_name,
                device=available_gpus[0],
                torch_dtype=torch.bfloat16,
                clean_up_tokenization_spaces=False,
                trust_remote_code=True
            )
        except RuntimeError:
            logging.error("Error loading the model.")
            raise RuntimeError("Error loading the model.")
        # TODO: implement dynamic batching
        # request_queue = Queue()
        # response_dict = {}
        lock = Lock()

        # def process_batch():
        #     while True:
        #         batch = []
        #         key = None
        #         batch_ids = []
        #         while len(batch) < 8 and not request_queue.empty():
        #             request_id, data = request_queue.queue[0]
        #             new_key = (data.get("temperature", 0.9), data.get("max_new_tokens", 8192))
        #             if key is None:
        #                 key = new_key
        #             elif key != new_key:
        #                 break
        #             request_id, data = request_queue.get()
        #             batch.append(data)
        #             batch_ids.append(request_id)
                
        #         if batch:
        #             temperature, max_new_tokens = key
        #             input_texts = [item["input_text"] for item in batch]

        #             responses = model(
        #                 input_texts,
        #                 temperature=temperature,
        #                 max_new_tokens=max_new_tokens
        #             )
        #             print("here2")

        #             with lock:
        #                 for i, response in enumerate(responses):
        #                     response_dict[batch_ids[i]] = response
        #                     request_queue.task_done()

        # Thread(target=process_batch, daemon=True).start()

        @app.route("/infer", methods=["POST"])
        def infer():
            data = request.get_json()
            input_text = data.get("input_text", "")
            temperature = data.get("temperature", 0.9)
            max_new_tokens = data.get("max_new_tokens", 1024)
            if not input_text:
                logging.error("Missing input_text in request")
                return jsonify({"error": "Missing input_text"}), 400
            
            with lock:
                with torch.no_grad():
                    response = model(input_text, temperature=temperature, max_new_tokens=max_new_tokens)
                torch.cuda.empty_cache()

            # request_id = id(data)
            # with lock:
            #     response_dict[request_id] = None
            # request_queue.put((request_id, data))

            # while True:
            #     with lock:
            #         if response_dict[request_id] is not None:
            #             response = response_dict.pop(request_id)
            #             break

            logging.info(f"Inference completed for input: {input_text}")
            return jsonify(response), 200

    def start_sent_trans_server(self, app: Flask, available_gpus: list[int]):
        model = SentenceTransformer(self.model_name, device=f"cuda:{available_gpus[0]}", trust_remote_code=True)
        model.eval()
        logging.info(f"Loaded SentenceTransformer model {self.model_name}")
        
        # TODO: implement dynamic batching
        # request_queue = Queue()
        # response_dict = {}
        lock = Lock()

        # def process_batch():
        #     while True:
        #         batch = []
        #         batch_ids = []
        #         while len(batch) < 8 and not request_queue.empty():
        #             request_id, data = request_queue.get()
        #             batch.append(data)
        #             batch_ids.append(request_id)
                
        #         if batch:
        #             input_texts = [item["input_text"] for item in batch]
        #             embeddings = model.encode(input_texts)

        #             with lock:
        #                 for i, embedding in enumerate(embeddings):
        #                     response_dict[batch_ids[i]] = embedding
        #                     request_queue.task_done()

        # Thread(target=process_batch, daemon=True).start()

        @app.route("/infer", methods=["POST"])
        def infer():
            data = request.get_json()
            input_text = data.get("input_text", "")
            if not input_text:
                logging.error("Missing input_text in request")
                return jsonify({"error": "Missing input_text"}), 400
            
            with lock:
                with torch.no_grad():
                    response = model.encode(input_text)
                torch.cuda.empty_cache()

            # request_id = id(data)
            # with lock:
            #     response_dict[request_id] = None
            # request_queue.put((request_id, data))

            # while True:
            #     with lock:
            #         if response_dict[request_id] is not None:
            #             response = response_dict.pop(request_id)
            #             break

            logging.info(f"Inference completed for input: {input_text}")
            return jsonify({"embeddings": response.tolist()}), 200
        
    def start_openai_server(self, app: Flask):
        @app.route("/infer", methods=["POST"])
        def infer():
            data = request.get_json()
            input_text = data.get("input_text", "")
            api_key = data.get("api_key", "")
            temperature = data.get("temperature", 0.9)
            max_new_tokens = data.get("max_new_tokens", 8192)
            if not input_text or not api_key:
                logging.error("Missing input_text or api_key in request")
                return jsonify({"error": "Missing input_text or api_key"}), 400
            
            openai_client = OpenAI(api_key=api_key)
            response = [{"generated_text":[{"content": openai_client.chat.completions.create(
                model=self.model_name,
                messages=input_text,
                temperature=temperature,
                max_completion_tokens=max_new_tokens
            ).choices[0].message.content}]}]
            logging.info(f"Inference completed for input: {input_text}")
            return jsonify(response), 200
    
    def start_genai_server(self, app: Flask):
        @app.route("/infer", methods=["POST"])
        def infer():
            data = request.get_json()
            input_text = data.get("input_text", "")
            api_key = data.get("api_key", "")
            temperature = data.get("temperature", 0.9)
            max_new_tokens = data.get("max_new_tokens", 8192)
            if not input_text or not api_key:
                logging.error("Missing input_text or api_key in request")
                return jsonify({"error": "Missing input_text or api_key"}), 400
            
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(self.model_name, generation_config={"temperature": temperature, "max_output_tokens": max_new_tokens})
            response = [{"generated_text":[{"content": client.generate_content("\n".join([item["content"] for item in input_text])).text}]}]
            logging.info(f"Inference completed for input: {input_text}")
            return jsonify(response), 200
        
    def start_ollama_server(self, app: Flask):
        # TODO: Implement Ollama server
        pass