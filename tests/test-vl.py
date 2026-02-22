# Use a pipeline as a high-level helper
import requests
# from transformers import pipeline
# Define the URL
url = "http://localhost:1121/infer"

payload = {
    "payload": {
        "inputs": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                    {"type": "text", "text": "What animal is on the candy?"}
                ]
            }
        ],
        "max_new_tokens": 1024
    },
    "type": "image-text-to-text",
    "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
    "timeout": 0
}

response = requests.post(url, json=payload)
response.raise_for_status()
result = response.json().get("response", "No response field")
print(result)
    # print(f"[Thread {thread_id}] Success: {result[:60]}...")

    # print(f"[Thread {thread_id}] Error: {e}")

# pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-32B-Instruct")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     },
# ]
# pipe(text=messages)


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-32B-Instruct")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     },
# ]
# print(pipe(text=messages))