#!/usr/bin/env python3
"""
Simple Claude API test - single request.
"""

import requests
import json

# Simple test payload
payload = {
    "type": "claude",
    "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "payload": {
        "messages": [
            {"role": "user", "content": "Hello Claude! Tell me a fun fact about Python programming."}
        ],
        "max_tokens": 4000,
        "temperature": 0.7
    },
    "timeout": 60,
    "max_retries": 3
}

print("Sending request to Claude API...")
print(f"Prompt: {payload['payload']['messages'][0]['content']}\n")

try:
    response = requests.post("http://localhost:1121/infer", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✓ Success!")
        print(f"\nClaude's response:\n{result['response']}")
    else:
        print(f"✗ Error {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.ConnectionError:
    print("✗ Connection Error: Cannot connect to ModelServer at http://localhost:1121")
    print("Please ensure the ModelServer is running.")
except Exception as e:
    print(f"✗ Exception: {str(e)}")
