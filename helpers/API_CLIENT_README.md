# Model Server API Client

A Python client library for interacting with the Model Server, supporting various AI models including OpenAI embeddings, chat completions, Claude, sentence transformers, and more.

## Installation

```bash
pip install requests numpy
```

## Quick Start

```python
from api_client import ModelServerClient

# Create a client
client = ModelServerClient(base_url="http://localhost:1121")

# Create embeddings
embeddings = client.create_embeddings(["Hello world", "How are you?"])
print(f"Created {len(embeddings)} embeddings")

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]
response = client.chat_completion(messages)
print(response)

# Close the client
client.close()
```

## Using Context Manager

```python
from api_client import ModelServerClient

with ModelServerClient() as client:
    embeddings = client.create_embeddings(["Test sentence"])
    print(embeddings)
# Client is automatically closed
```

## API Reference

### Initialization

```python
client = ModelServerClient(
    base_url="http://localhost:1121",  # Model server URL
    timeout=300                         # Default timeout in seconds
)
```

### OpenAI Embeddings

Create embeddings using Azure OpenAI's embedding models.

```python
embeddings = client.create_embeddings(
    input_texts=["text1", "text2"],           # List of texts or single text
    model="text-embedding-3-small",           # Model name
    timeout=300                                # Optional timeout
)
# Returns: List[List[float]] - list of embedding vectors
```

**Example:**
```python
texts = ["Python is great", "I love programming"]
embeddings = client.create_embeddings(texts)

# embeddings[0] = [0.123, -0.456, 0.789, ...]  # First embedding
# embeddings[1] = [0.234, -0.567, 0.890, ...]  # Second embedding
```

### Chat Completion (OpenAI)

Generate chat completions using Azure OpenAI.

```python
response = client.chat_completion(
    messages=[                                 # Conversation history
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    model="gpt-4.1-mini",                     # Model name
    temperature=0.7,                           # Sampling temperature (0-2)
    max_tokens=None,                           # Max tokens in response
    timeout=300                                # Optional timeout
)
# Returns: str - generated response
```

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Explain list comprehension in Python"}
]
response = client.chat_completion(messages, temperature=0.7)
print(response)
```

### Sentence Transformers

Encode sentences using local sentence transformer models.

```python
embeddings = client.encode_sentences(
    sentences=["sentence1", "sentence2"],                    # List or single sentence
    model="sentence-transformers/all-MiniLM-L6-v2",         # Model name
    normalize_embeddings=False,                              # Whether to normalize
    timeout=300                                              # Optional timeout
)
# Returns: List[List[float]] or List[float] - embeddings
```

**Example:**
```python
sentences = [
    "The cat sits on the mat",
    "A dog plays in the park"
]
embeddings = client.encode_sentences(
    sentences,
    model="sentence-transformers/all-MiniLM-L6-v2",
    normalize_embeddings=True
)
```

### Text Generation

Generate text using local transformer models.

```python
response = client.generate_text(
    messages=[{"role": "user", "content": "..."}],  # Single conversation
    # OR
    messages=[[...], [...]],                        # Batch of conversations
    model="Qwen/Qwen2.5-14B-Instruct",             # Model name
    max_new_tokens=512,                             # Max tokens to generate
    temperature=0.7,                                # Sampling temperature
    top_p=0.9,                                      # Nucleus sampling
    timeout=300                                     # Optional timeout
)
# Returns: str or List[str] - generated text(s)
```

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
]
response = client.generate_text(
    messages,
    model="Qwen/Qwen2.5-14B-Instruct",
    max_new_tokens=256
)
```

### Claude (via AWS Bedrock)

Generate completions using Claude models.

```python
response = client.claude_completion(
    messages=[{"role": "user", "content": "..."}],        # Conversation
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Claude model ID
    max_tokens=4000,                                      # Max tokens
    temperature=0.1,                                      # Temperature
    timeout=300                                           # Optional timeout
)
# Returns: str - generated response
```

**Example:**
```python
messages = [
    {"role": "user", "content": "Explain quantum entanglement in simple terms"}
]
response = client.claude_completion(messages, max_tokens=500)
```

### CLIP Features

Extract features using CLIP models.

**Image Features:**
```python
features = client.get_image_features(
    images=["path/to/image1.jpg", "path/to/image2.jpg"],  # Image paths
    model="openai/clip-vit-base-patch32",                  # CLIP model
    timeout=300                                            # Optional timeout
)
# Returns: List[List[float]] - image feature vectors
```

**Text Features:**
```python
features = client.get_text_features(
    texts=["a photo of a cat", "a photo of a dog"],       # Text descriptions
    model="openai/clip-vit-base-patch32",                  # CLIP model
    timeout=300                                            # Optional timeout
)
# Returns: List[List[float]] - text feature vectors
```

## Examples

### 1. Semantic Search

```python
from api_client import ModelServerClient
import numpy as np

client = ModelServerClient()

# Documents
documents = [
    "Python is a programming language",
    "Machine learning uses neural networks",
    "JavaScript is used for web development"
]

# Query
query = "Tell me about AI"

# Get embeddings
all_texts = documents + [query]
embeddings = client.create_embeddings(all_texts)

# Compute similarities
query_emb = np.array(embeddings[-1])
for i, doc_emb in enumerate(embeddings[:-1]):
    similarity = np.dot(query_emb, doc_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
    )
    print(f"{documents[i]}: {similarity:.4f}")

client.close()
```

### 2. RAG (Retrieval-Augmented Generation)

```python
from api_client import ModelServerClient

client = ModelServerClient()

# Step 1: Retrieve relevant documents
knowledge_base = [
    "Python was created by Guido van Rossum in 1991",
    "The Eiffel Tower is in Paris, France",
    "Machine learning is a subset of AI"
]

query = "When was Python created?"
embeddings = client.create_embeddings(knowledge_base + [query])

# Find most similar document (simplified)
# ... similarity computation ...
context = knowledge_base[0]  # Most relevant

# Step 2: Generate answer with context
messages = [
    {"role": "system", "content": "Answer based on the context."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]

answer = client.chat_completion(messages)
print(answer)

client.close()
```

### 3. Batch Processing

```python
from api_client import ModelServerClient

client = ModelServerClient()

# Process many texts at once
texts = [f"Document {i}" for i in range(100)]

# Single API call for all texts
embeddings = client.create_embeddings(texts)
print(f"Processed {len(embeddings)} documents")

client.close()
```

### 4. Multi-turn Conversation

```python
from api_client import ModelServerClient

client = ModelServerClient()

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Turn 1
messages.append({"role": "user", "content": "What is Python?"})
response = client.chat_completion(messages)
messages.append({"role": "assistant", "content": response})
print(f"Assistant: {response}")

# Turn 2
messages.append({"role": "user", "content": "What are its main features?"})
response = client.chat_completion(messages)
print(f"Assistant: {response}")

client.close()
```

## Running Examples

The `examples.py` file contains comprehensive examples:

```bash
python examples.py
```

This will run:
- Embeddings and similarity computation
- Chat completions
- Semantic search
- Batch processing
- RAG pipeline
- Sentiment analysis

## Error Handling

```python
from api_client import ModelServerClient
import requests

client = ModelServerClient()

try:
    embeddings = client.create_embeddings(["test"])
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Could not connect to server")
except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
```

## Supported Model Types

| Type | Description | Example Models |
|------|-------------|----------------|
| `openai-embedding` | Azure OpenAI embeddings | `text-embedding-3-small` |
| `openai` | Azure OpenAI chat | `gpt-4.1-mini` |
| `claude` | Claude via AWS Bedrock | `claude-sonnet-4` |
| `sent-trans` | Sentence transformers | `all-MiniLM-L6-v2` |
| `text-generation` | Local transformers | `Qwen2.5-14B-Instruct` |
| `clip` | CLIP models | `clip-vit-base-patch32` |

## Configuration

### Environment Variables

If needed, you can configure API keys in a `.env` file:

```bash
OPENAI_API_KEYS="key1 key2 key3"
GEMINI_API_KEYS="key1 key2"
OLLAMA_PORTS="6001 6002"
```

### Timeouts

Adjust timeouts based on your use case:

```python
# Short timeout for quick operations
client = ModelServerClient(timeout=30)

# Long timeout for heavy operations
client = ModelServerClient(timeout=600)

# Per-request timeout
embeddings = client.create_embeddings(texts, timeout=60)
```

## Best Practices

1. **Use context managers** to ensure proper cleanup:
   ```python
   with ModelServerClient() as client:
       result = client.create_embeddings(["test"])
   ```

2. **Batch requests** when possible for better performance:
   ```python
   # Good: Single request for multiple texts
   embeddings = client.create_embeddings(["text1", "text2", "text3"])

   # Avoid: Multiple requests
   emb1 = client.create_embeddings(["text1"])
   emb2 = client.create_embeddings(["text2"])
   ```

3. **Handle errors gracefully**:
   ```python
   try:
       result = client.chat_completion(messages)
   except Exception as e:
       logger.error(f"Request failed: {e}")
       # Implement retry logic or fallback
   ```

4. **Set appropriate timeouts** based on model and payload size

5. **Reuse the client instance** for multiple requests instead of creating new ones

## Troubleshooting

### Connection refused
```
Error: Connection refused
```
**Solution:** Ensure the model server is running on `http://localhost:1121`

### Timeout errors
```
Error: Request timed out
```
**Solution:** Increase timeout or check server load

### Server errors
```
Server error: Max retries exceeded
```
**Solution:** Check server logs and ensure the model is loaded correctly

## License

MIT License
