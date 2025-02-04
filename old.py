from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, Pipeline
import google.generativeai as genai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import threading
import time
import logging
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global dictionary to keep track of loaded models and their usage
loaded_models = {}
# Lock for thread safety
model_lock = threading.Lock()
# Idle timeout in seconds
IDLE_TIMEOUT = 600

OpenAI_KEY = ""
GENAI_KEY = ""

genai.configure(api_key=GENAI_KEY)
openai_client = OpenAI(api_key=OpenAI_KEY)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.last_used = {}

    def get_sorted_gpus_by_memory(self):
        """
        Get a list of GPU indices sorted by available memory in descending order.
        """
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free_memory = torch.cuda.mem_get_info()[0]  # Free memory in bytes
            gpu_memory.append((i, free_memory))
        # Sort GPUs by free memory in descending order
        sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
        return [gpu[0] for gpu in sorted_gpus]

    def load_model(self, type, model_name):
        """
        Load a model of the specified type and name. If the model is already loaded, return it.
        """
        with model_lock:
            if model_name not in self.models:
                logger.info(f"Loading model: {model_name}")
                # Get sorted GPUs by available memory
                available_gpus = self.get_sorted_gpus_by_memory()
                if not available_gpus:
                    raise RuntimeError("No available GPUs to load the model.")
                
                try:
                    if type == "text-generation":
                        # Try loading the model on the GPU with the most free memory
                        try:
                            self.models[model_name] = pipeline(
                                "text-generation",
                                torch_dtype=torch.bfloat16,
                                model=model_name,
                                device=available_gpus[0],
                                clean_up_tokenization_spaces=False,
                                trust_remote_code=True
                            )
                        except RuntimeError:
                            logger.warning(f"Model {model_name} could not fit on a single GPU. Splitting across GPUs.")
                            # Load model across multiple GPUs (model parallelism)
                            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                            tokenizer = AutoTokenizer.from_pretrained(model_name)
                            device_map = "balanced"
                            self.models[model_name] = pipeline(
                                "text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                device_map=device_map,
                                torch_dtype=torch.bfloat16,
                                clean_up_tokenization_spaces=False,
                                trust_remote_code=True
                            )
                    elif type == "sent-trans":
                        self.models[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
                        self.models[model_name].eval()
                    else:
                        raise ValueError("Invalid model type. Supported types: text-generation, embedding")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

            self.last_used[model_name] = time.time()
        return self.models[model_name]

    def get_model(self, type, model_name):
        """
        Get a model of the specified type and name. If the model is not loaded, load it.
        """
        with model_lock:
            if model_name in self.models:
                self.last_used[model_name] = time.time()
                return self.models[model_name]
        return self.load_model(type, model_name)

    def unload_cuda_context(self):
        """
        Unload the CUDA context to free up GPU memory.
        """
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def offload_idle_models(self):
        """
        Offload models that have been idle for longer than the IDLE_TIMEOUT.
        """
        while True:
            with model_lock:
                current_time = time.time()
                to_offload = [
                    model_name
                    for model_name, last_used_time in self.last_used.items()
                    if current_time - last_used_time > IDLE_TIMEOUT
                ]
                for model_name in to_offload:
                    logger.info(f"Offloading model: {model_name}")
                    model = self.models[model_name]
                    if isinstance(model, SentenceTransformer):
                        del self.models[model_name]
                    elif isinstance(model, Pipeline):
                        model.model.to('cpu')
                        del self.models[model_name]
                    else:
                        del self.models[model_name]
                    del self.last_used[model_name]
                self.unload_cuda_context()
            time.sleep(60)

manager = ModelManager()

@app.route("/infer", methods=["POST"])
def query_model():
    """
    Handle POST requests to query a model. The request must contain 'model_name', 'query', and 'type'.
    """
    data = request.json
    if not data or "model_name" not in data or "input_text" not in data or "type" not in data:
        return jsonify({"error": "Invalid request"}), 400

    model_name = data["model_name"]
    type = data.get("type", "text-generation")
    input_text = data["input_text"]
    temperature = data.get("temperature", 0.9)
    max_new_tokens = data.get("max_new_tokens", 8192)

    try:
        if type == "text-generation":
            if model_name in ["gpt-4o-mini"]:
                response = [{"generated_text":[{"content": openai_client.chat.completions.create(
                    model=model_name,
                    messages=input_text,
                    temperature=temperature,
                    max_completion_tokens=max_new_tokens
                ).choices[0].message.content}]}]
            elif model_name in ["gemini-1.5-flash"]:
                time.sleep(10)
                client = genai.GenerativeModel(model_name)
                response = [{"generated_text":[{"content": client.generate_content("\n".join([item["content"] for item in input_text])).text}]}]
            else:
                model = manager.get_model(type, model_name)
                response = model(input_text, temperature=temperature, max_new_tokens=max_new_tokens)
        elif type == "sent-trans":
            model = manager.get_model(type, model_name)
            embeddings = model.encode(input_text)
            response = {"embeddings": embeddings.tolist()}
        else:
            raise ValueError("Invalid model type. Supported types: text-generation, embedding")
        torch.cuda.empty_cache()
        return jsonify(response)
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        logger.error(f"RuntimeError: {str(re)}")
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start a background thread to offload idle models
    threading.Thread(target=manager.offload_idle_models, daemon=True).start()
    logger.info("Starting the application...")
    app.run(host="0.0.0.0", port=5000)
