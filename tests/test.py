import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# Define the URL
url = "http://localhost:1121/infer"

# Define the function to send a POST request with unique content
def send_request(thread_id: int):
    custom_content = f"Tell me about superhero number {thread_id}"
    payload: Dict[str, Any] = {
        "payload": {
            "inputs": custom_content,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 50,
            "max_tokens": 2048,
            "repetition_penalty": 1.05
        },
        "type": "vllm-text-generation",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "timeout": 0
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json().get("response", "No response field")
        print(f"[Thread {thread_id}] Success: {result[:60]}...")
    except Exception as e:
        print(f"[Thread {thread_id}] Error: {e}")

# Run 100 threads
def main():
    num_threads = 50
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_threads)]
        for _ in as_completed(futures):
            pass  # Output already handled in send_request

if __name__ == "__main__":
    main()


# import requests

# # payload = {
# #     "payload": {
# #         "inputs": [[
# #             {"role": "user", "content": "Tell me about captain america"}
# #         ],
# #                      [
# #             {"role": "user", "content": "Tell me about iron man"}
# #         ]],
# #         "temperature": 0.9,
# #         "max_new_tokens": 1024
# #     },
# #     "type": "text-generation",
# #     "model_name": "Qwen/Qwen2.5-14B-Instruct"
# # }

# ocr_prompt = """You are an expert OCR specialist. Analyze this document image and extract ALL text while preserving the EXACT structure, formatting, and layout.

# CRITICAL REQUIREMENTS:
# 1. Maintain exact spacing, indentation, and line breaks
# 2. Preserve tables, lists, headers, and hierarchical structure  
# 3. Keep mathematical formulas, equations, and special characters intact
# 4. Maintain bullet points, numbering, and formatting elements
# 5. Preserve any diagrams or flowchart text labels
# 6. Do NOT summarize, paraphrase, or interpret - extract verbatim text only
# 7. If text is unclear, indicate with [UNCLEAR: best_guess] but still attempt extraction
# 8. Maintain reading order (left-to-right, top-to-bottom for most languages)

# FORMAT YOUR RESPONSE AS:
# Pure text extraction maintaining original structure
# Use markdown formatting only where it helps preserve structure (tables, headers, lists)  
# Indicate page/section breaks where visible
# Preserve any footer/header information

# Extract everything visible in the image with maximum accuracy and structural fidelity. Respond with only the extracted text, no additional commentary."""

# import base64

# with open("Screenshot 2025-04-25 133322.png", "rb") as f:
#     base64_image = base64.b64encode(f.read()).decode('utf-8')

# payload = {
#     "payload": {
#         "inputs": [
#             {
#                 "role": "user", 
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": ocr_prompt
#                     },
#                     {
#                         "type": "image",
#                         "image": f"data:image/png;base64,{base64_image}"
#                     }
#                 ]
#             }
#         ],
#         "max_new_tokens": 4096,
#         "temperature": 0.1
#     },
#     "timeout":0,
#     "type": "image-text-to-text",
#     "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"
# }

# response = requests.post("http://localhost:1121/infer", json=payload)
# response.raise_for_status()
# print(response.json()["response"])


# from concurrent.futures import ThreadPoolExecutor, as_completed
# import requests
# import logging

# logger = logging.basicConfig()


# def model_inference_node(self, state: AgentState) -> AgentState:
#     """Call the model inference with batch format"""
#     def inferModel(payload):
#         response = requests.post(
#                             self.inference_endpoint,
#                             json=payload,
#                             timeout=180
#                         )
#         response.raise_for_status()
#         return response.json()["response"]
#     try:
#         logger.info("Running batch model inference")
#         # batch_inputs = []            
#         if state.get("crawled_data"):
#             with ThreadPoolExecutor() as executor:
#                 futures = []
#                 for url, data in list(state["crawled_data"].items()):  
#                     if data.get("html"):
#                         prompt = f"""
#     Extract product information from this webpage:
#     URL: {url}

#     Look for:
#     - Product names
#     - Prices (original and discounted)
#     - Availability
#     - Ratings
#     - Quantity/Weight
#     - Brand
#     - Category

#     Provide the information in JSON format.
#     """
#                         payload = {
#                             "payload": {
#                                 "inputs": [
#                                         {"role": "system", "content": prompt},
#                                         {"role":"user", "content":data["html"]}
#                                     ],
#                                 "temperature": 0.7,
#                                 "max_new_tokens": 1024
#                             },
#                             "type": "text-generation",
#                             "model_name": "Qwen/Qwen2.5-14B-Instruct",
#                             "timeout": 0
#                         }
                        
#                         logger.info(f"Sending {len(prompt)} queries to model inference")
#                         futures.append(executor.submit(inferModel, payload))
                        
#                         for future in as_completed(futures):
#                             try:
#                                 result = future.result()
#                                 state
#                             except Exception:
#                                 continue
                            
                        
#                         if response.status_code == 200:
#                             result = response.json()
#                             state["model_inference"] = result
#                             logger.info(f"Model inference completed successfully with {len(result)} results")
#                         else:
#                             logger.error(f"Model inference failed with status: {response.status_code}")
#                             logger.error(f"Response: {response.text[:500]}")
#                             state["model_inference"] = None
                    
#     except Exception as e:
#         logger.error(f"Error in model inference node: {e}")
#         state["model_inference"] = None
        
#     return state