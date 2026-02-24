#!/usr/bin/env python3
"""
Test script for Claude API endpoint in ModelServer.

This script sends test requests to the ModelServer's Claude endpoint
and displays the responses.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
MODEL_SERVER_URL = "http://localhost:1121/infer"
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def test_simple_request():
    """Test a simple text request to Claude."""
    print("=" * 60)
    print("Test 1: Simple Text Request")
    print("=" * 60)

    payload = {
        "type": "claude",
        "model_name": CLAUDE_MODEL_ID,
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please introduce yourself in one sentence.",
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        },
        "timeout": 60,
        "max_retries": 3,
    }

    print(f"Sending request: {json.dumps(payload, indent=2)}")
    print("\nWaiting for response...")

    start_time = time.time()
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload)
        elapsed_time = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse: {result.get('response', 'No response field')}")
        else:
            print(f"\nError: {response.text}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

    print("\n")


def test_conversation():
    """Test a multi-turn conversation with Claude."""
    print("=" * 60)
    print("Test 2: Multi-turn Conversation")
    print("=" * 60)

    payload = {
        "type": "claude",
        "model_name": CLAUDE_MODEL_ID,
        "payload": {
            "messages": [
                {"role": "user", "content": "What is 15 + 27?"},
                {"role": "assistant", "content": "15 + 27 = 42"},
                {"role": "user", "content": "Now multiply that by 3"},
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        },
        "timeout": 60,
        "max_retries": 3,
    }

    print(
        f"Sending conversation: {json.dumps(payload['payload']['messages'], indent=2)}"
    )
    print("\nWaiting for response...")

    start_time = time.time()
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload)
        elapsed_time = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\nClaude's Response: {result.get('response', 'No response field')}")
        else:
            print(f"\nError: {response.text}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

    print("\n")


def test_longer_generation():
    """Test a request that requires longer generation."""
    print("=" * 60)
    print("Test 3: Longer Generation")
    print("=" * 60)

    payload = {
        "type": "claude",
        "model_name": CLAUDE_MODEL_ID,
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short haiku about artificial intelligence.",
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        },
        "timeout": 60,
        "max_retries": 3,
    }

    print(f"Prompt: {payload['payload']['messages'][0]['content']}")
    print("\nWaiting for response...")

    start_time = time.time()
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload)
        elapsed_time = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\nHaiku:\n{result.get('response', 'No response field')}")
        else:
            print(f"\nError: {response.text}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

    print("\n")


def test_with_higher_retries():
    """Test request with custom max_retries."""
    print("=" * 60)
    print("Test 4: Request with Higher Max Retries")
    print("=" * 60)

    payload = {
        "type": "claude",
        "model_name": CLAUDE_MODEL_ID,
        "payload": {
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 50,
            "temperature": 0.1,
        },
        "timeout": 60,
        "max_retries": 5,  # Higher than default
    }

    print(f"Max retries set to: {payload['max_retries']}")
    print(f"Prompt: {payload['payload']['messages'][0]['content']}")
    print("\nWaiting for response...")

    start_time = time.time()
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload)
        elapsed_time = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse: {result.get('response', 'No response field')}")
        else:
            print(f"\nError: {response.text}")

    except Exception as e:
        print(f"\nException occurred: {str(e)}")

    print("\n")


def test_parallel_requests():
    """Test multiple parallel requests to see load balancing."""
    print("=" * 60)
    print("Test 5: Parallel Requests (5 concurrent)")
    print("=" * 60)

    import concurrent.futures

    def send_request(request_id: int) -> Dict[str, Any]:
        payload = {
            "type": "claude",
            "model_name": CLAUDE_MODEL_ID,
            "payload": {
                "messages": [
                    {"role": "user", "content": f"Say 'Request {request_id} completed'"}
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            },
            "timeout": 60,
            "max_retries": 3,
        }

        start_time = time.time()
        response = requests.post(MODEL_SERVER_URL, json=payload)
        elapsed_time = time.time() - start_time

        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "elapsed_time": elapsed_time,
            "response": (
                response.json() if response.status_code == 200 else response.text
            ),
        }

    print("Sending 5 parallel requests...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_request, i) for i in range(1, 6)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    total_time = time.time() - start_time

    print(f"\nAll requests completed in {total_time:.2f}s\n")

    for result in sorted(results, key=lambda x: x["request_id"]):
        print(f"Request {result['request_id']}:")
        print(f"  Status: {result['status_code']}")
        print(f"  Time: {result['elapsed_time']:.2f}s")
        if result["status_code"] == 200:
            print(f"  Response: {result['response'].get('response', 'N/A')}")
        else:
            print(f"  Error: {result['response']}")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Claude API Test Suite")
    print("=" * 60)
    print(f"Target: {MODEL_SERVER_URL}")
    print(f"Model: {CLAUDE_MODEL_ID}")
    print("=" * 60 + "\n")

    # Check if server is running
    try:
        response = requests.get("http://localhost:1121/")
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to ModelServer at http://localhost:1121")
        print("Please ensure the ModelServer is running.")
        return

    # Run tests
    tests = [
        test_simple_request,
        test_conversation,
        test_longer_generation,
        test_with_higher_retries,
        test_parallel_requests,
    ]

    for i, test in enumerate(tests, 1):
        try:
            test()
            time.sleep(1)  # Small delay between tests
        except KeyboardInterrupt:
            print("\n\nTests interrupted by user.")
            break
        except Exception as e:
            print(f"\nTest {i} failed with exception: {str(e)}\n")

    print("=" * 60)
    print("Test Suite Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
