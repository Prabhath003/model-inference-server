# Claude File Upload Guide

Guide for uploading files (documents, images) to Claude API via ModelServer.

## Supported File Types

- ✅ **PDF Documents** - Base64 encoded
- ✅ **Images** - PNG, JPEG, WebP, GIF (Base64 encoded)
- ✅ **Text Content** - Direct inclusion in messages

## Usage Examples

### 1. Text File Summarization

For text files (Markdown, TXT, etc.), include content directly:

```python
import requests

with open('document.md', 'r') as f:
    content = f.read()

payload = {
    "type": "claude",
    "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "payload": {
        "messages": [
            {
                "role": "user",
                "content": f"Please summarize this document:\n\n{content}"
            }
        ],
        "max_tokens": 1000
    }
}

response = requests.post("http://localhost:1121/infer", json=payload)
```

### 2. PDF Upload

For PDF files, use base64 encoding with document type:

```python
import requests
import base64

# Encode PDF to base64
with open('document.pdf', 'rb') as f:
    pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

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
                            "data": pdf_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please analyze this PDF and provide key insights."
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }
}

response = requests.post("http://localhost:1121/infer", json=payload)
```

### 3. Image Upload

For image files:

```python
import requests
import base64

# Encode image to base64
with open('image.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "type": "claude",
    "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "payload": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
}

response = requests.post("http://localhost:1121/infer", json=payload)
```

### 4. Multiple Files

You can include multiple documents/images in one request:

```python
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
                            "data": pdf1_base64
                        }
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf2_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Compare these two documents and highlight the differences."
                    }
                ]
            }
        ],
        "max_tokens": 3000
    }
}
```

## Content Format Specifications

### Document Structure

```python
{
    "type": "document",
    "source": {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "<base64_encoded_pdf>"
    }
}
```

### Image Structure

```python
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",  # or "image/jpeg", "image/webp", "image/gif"
        "data": "<base64_encoded_image>"
    }
}
```

### Text Structure

```python
{
    "type": "text",
    "text": "Your text content here"
}
```

## Supported Media Types

### Documents
- `application/pdf`

### Images
- `image/png`
- `image/jpeg`
- `image/webp`
- `image/gif`

## Test Scripts

### Quick Text File Test
```bash
python test_claude_file_upload.py
```

This will:
1. Read the TEST_CLAUDE_README.md file
2. Send it to Claude for summarization
3. Display the summary

### Custom File Test

Modify `test_claude_file_upload.py` to test your own files:

```python
# For text files
test_file_summarization("path/to/your/file.md")

# For PDFs
test_pdf_upload("path/to/your/document.pdf")
```

## Size Limits

- **PDF**: Recommended max ~10MB (depends on AWS limits)
- **Images**: Recommended max ~5MB per image
- **Text**: Included in token limit (max_tokens)
- **Total Request**: Consider overall payload size

## Best Practices

1. **Compress Large PDFs**: Use PDF compression before upload
2. **Optimize Images**: Resize/compress images to reduce payload
3. **Batch Related Files**: Include related documents in one request
4. **Clear Instructions**: Always include clear text instructions with files
5. **Token Limits**: Account for document content in max_tokens setting

## Error Handling

### Base64 Encoding Errors
```python
try:
    with open(file_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
except Exception as e:
    print(f"Encoding error: {e}")
```

### File Size Errors
```python
import os

file_size = os.path.getsize(file_path)
if file_size > 10 * 1024 * 1024:  # 10MB
    print("File too large, consider compressing")
```

## Conversion Utilities

### Helper Function for File Encoding

```python
import base64
import mimetypes

def encode_file(file_path: str) -> dict:
    """Encode any file for Claude API."""

    # Determine media type
    mime_type, _ = mimetypes.guess_type(file_path)

    # Read and encode
    with open(file_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')

    # Determine content type
    if mime_type == 'application/pdf':
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data
            }
        }
    elif mime_type and mime_type.startswith('image/'):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data
            }
        }
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

# Usage
file_content = encode_file("document.pdf")
```

## Example Use Cases

### 📄 Document Analysis
- Contract review
- Report summarization
- Research paper analysis

### 🖼️ Image Analysis
- Chart/graph interpretation
- Screenshot analysis
- Visual content description

### 📊 Multi-modal Analysis
- Presentation slides (PDF) with charts
- Documents with embedded images
- Comparative document analysis

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Base64 decode error** | Ensure proper encoding without newlines |
| **File too large** | Compress or split the document |
| **Unsupported format** | Convert to PDF or supported image format |
| **Timeout** | Increase timeout for large documents |
| **Empty response** | Check if file is properly encoded |

## Performance Tips

1. **Pre-process Documents**: Extract text if only text is needed
2. **Optimize Images**: Use appropriate resolution (don't over-sample)
3. **Parallel Processing**: Send multiple documents in parallel requests
4. **Cache Results**: Store summaries to avoid re-processing
5. **Stream Large Files**: Consider chunking very large documents

## Response Format

Successful response:
```json
{
    "response": "Document summary: This is a test document about..."
}
```

Error response:
```json
{
    "error": "Failed to get response"
}
```
