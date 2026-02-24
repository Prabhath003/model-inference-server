import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct-1M")
messages = [
    {"role": "user", "content": "Who are you?"},
]
print(pipe(messages))
time.sleep(60)
