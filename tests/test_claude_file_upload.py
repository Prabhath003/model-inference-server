#!/usr/bin/env python3
"""
Claude API test with file upload - test document summarization.
"""

import requests
import json
import base64
import os


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_file_summarization(file_path: str):
    """Test Claude with a file upload for summarization."""

    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return

    # Read file content
    with open(file_path, "r") as f:
        file_content = f.read()

    print(f"Testing file upload and summarization")
    print(f"File: {file_path}")
    print(f"File size: {len(file_content)} characters\n")

    # For text files, we can include them directly in the message
    # For PDFs, we would need to base64 encode them

    payload = {
        "type": "claude",
        "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": f"""I'm sharing a document with you. Please read it and provide a concise summary.

Document content:
{file_content}

Please provide:
1. A brief summary (2-3 sentences)
2. Key points or highlights
3. Main purpose of the document""",
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        "timeout": 120,
        "max_retries": 3,
    }

    print("Sending request to Claude API...")
    print("Waiting for response...\n")

    try:
        response = requests.post("http://localhost:1121/infer", json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print("\n" + "=" * 60)
            print("Claude's Summary:")
            print("=" * 60)
            print(result["response"])
            print("=" * 60)
        else:
            print(f"✗ Error {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "✗ Connection Error: Cannot connect to ModelServer at http://localhost:1121"
        )
        print("Please ensure the ModelServer is running.")
    except Exception as e:
        print(f"✗ Exception: {str(e)}")


def test_pdf_upload(pdf_path: str):
    """Test Claude with a PDF file upload."""

    if not os.path.exists(pdf_path):
        print(f"✗ Error: File not found: {pdf_path}")
        return

    print(f"Testing PDF upload and analysis")
    print(f"File: {pdf_path}\n")

    # Encode PDF to base64
    pdf_base64 = encode_file_to_base64(pdf_path)

    # Claude expects documents in a specific format
    payload = {
        "type": "claude",
        "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please summarize this PDF document. Provide the main points and key takeaways.",
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.3,
        },
        "timeout": 120,
        "max_retries": 3,
    }

    print("Sending PDF to Claude API...")
    print("Waiting for response...\n")

    try:
        response = requests.post("http://localhost:1121/infer", json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✓ Success!")
            print("\n" + "=" * 60)
            print("Claude's PDF Analysis:")
            print("=" * 60)
            print(result["response"])
            print("=" * 60)
        else:
            print(f"✗ Error {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "✗ Connection Error: Cannot connect to ModelServer at http://localhost:1121"
        )
        print("Please ensure the ModelServer is running.")
    except Exception as e:
        print(f"✗ Exception: {str(e)}")


def main():
    """Run file upload tests."""

    print("\n" + "=" * 60)
    print("Claude API File Upload Test")
    print("=" * 60 + "\n")

    # Test 1: Summarize the README file
    readme_path = "TEST_CLAUDE_README.md"
    if os.path.exists(readme_path):
        test_file_summarization(readme_path)
    else:
        print(f"README not found at {readme_path}, trying alternate path...")
        readme_path = "tests/TEST_CLAUDE_README.md"
        if os.path.exists(readme_path):
            test_file_summarization(readme_path)
        else:
            print("Could not find README file")

    print("\n" + "=" * 60)

    # Uncomment to test PDF upload if you have a PDF file
    pdf_path = "/home/prabhath/CG-Traversal/output/company_aulyra_design_private_limited/input/company_gist.txt.pdf"
    if os.path.exists(pdf_path):
        print("\n")
        test_pdf_upload(pdf_path)

    print("\nTest completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
