"""
API Client for Model Server

This module provides a convenient interface to interact with the model server,
supporting various model types including text generation, embeddings, and more.
"""

import requests
from typing import Any, Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelServerClient:
    """Client for interacting with the Model Server API."""

    def __init__(self, base_url: str = "http://localhost:1121", timeout: int = 300):
        """
        Initialize the Model Server Client.

        Args:
            base_url: Base URL of the model server (default: http://localhost:1121)
            timeout: Default timeout for requests in seconds (default: 300)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def _make_request(
        self,
        payload: Dict[str, Any],
        model_name: str,
        model_type: str,
        timeout: Optional[int] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make a request to the model server.

        Args:
            payload: The request payload specific to the model type
            model_name: Name of the model to use
            model_type: Type of the model (e.g., 'openai', 'openai-embedding', 'sent-trans')
            timeout: Request timeout in seconds (uses default if not specified)
            max_retries: Maximum number of retries on failure

        Returns:
            Response from the server

        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/infer"

        data = {
            "type": model_type,
            "model_name": model_name,
            "payload": payload,
            "timeout": timeout or self.timeout,
            "max_retries": max_retries
        }

        logger.info(f"Making request to {url} with type={model_type}, model={model_name}")

        try:
            response = self.session.post(url, json=data, timeout=timeout or self.timeout)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                logger.error(f"Server returned error: {result['error']}")
                raise Exception(f"Server error: {result['error']}")

            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    # ========== OpenAI Embedding Methods ==========

    def create_openai_embeddings(
        self,
        input_texts: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        timeout: Optional[int] = None
    ) -> List[List[float]]:
        """
        Create embeddings using Azure OpenAI embedding model.

        Args:
            input_texts: Single text string or list of text strings to embed
            model: Model name (default: "text-embedding-3-small")
            timeout: Request timeout in seconds

        Returns:
            List of embeddings (each embedding is a list of floats)

        Example:
            >>> client = ModelServerClient()
            >>> embeddings = client.create_embeddings(["Hello world", "How are you?"])
            >>> print(len(embeddings))  # 2
            >>> print(len(embeddings[0]))  # 1536 (embedding dimension)
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        payload: Dict[str, Any] = {
            "input": input_texts,
            "model": model
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="openai-embedding",
            timeout=timeout
        )

        return response["response"]

    # ========== OpenAI Chat Completion Methods ==========

    def openai_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Create a chat completion using Azure OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (default: "gpt-4.1-mini")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds

        Returns:
            Generated response text

        Example:
            >>> client = ModelServerClient()
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is the capital of France?"}
            ... ]
            >>> response = client.chat_completion(messages)
            >>> print(response)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="openai",
            timeout=timeout
        )

        return response["response"]

    # ========== Sentence Transformer Methods ==========

    def create_transformer_embeddings(
        self,
        sentences: Union[str, List[str]],
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = False,
        timeout: Optional[int] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode sentences using Sentence Transformers.

        Args:
            sentences: Single sentence or list of sentences
            model: Sentence transformer model name
            normalize_embeddings: Whether to normalize embeddings
            timeout: Request timeout in seconds

        Returns:
            Single embedding or list of embeddings

        Example:
            >>> client = ModelServerClient()
            >>> embeddings = client.encode_sentences(
            ...     ["This is a sentence", "This is another sentence"],
            ...     model="sentence-transformers/all-MiniLM-L6-v2"
            ... )
        """
        payload: Dict[str, Any] = {
            "sentences": sentences,
            "normalize_embeddings": normalize_embeddings
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="sent-trans",
            timeout=timeout
        )

        return response["response"]

    # ========== Text Generation Methods ==========

    def HF_chat_completion(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: Optional[int] = None
    ) -> Union[str, List[str]]:
        """
        Generate text using local text generation models.

        Args:
            messages: Single conversation or batch of conversations
            model: Model name
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            timeout: Request timeout in seconds

        Returns:
            Generated text or list of generated texts

        Example:
            >>> client = ModelServerClient()
            >>> messages = [
            ...     {"role": "user", "content": "What is AI?"}
            ... ]
            >>> response = client.generate_text(messages, model="Qwen/Qwen2.5-14B-Instruct")
        """
        payload: Dict[str, Any] = {
            "inputs": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="text-generation",
            timeout=timeout
        )

        return response["response"]

    # ========== Claude Methods ==========

    def claude_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        timeout: Optional[int] = None
    ) -> str:
        """
        Create a completion using Claude via AWS Bedrock.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Claude model ID
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            Generated response text

        Example:
            >>> client = ModelServerClient()
            >>> messages = [
            ...     {"role": "user", "content": "Explain quantum computing"}
            ... ]
            >>> response = client.claude_completion(messages)
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="claude",
            timeout=timeout
        )

        return response["response"]

    # ========== CLIP Methods ==========

    def get_image_features_HF(
        self,
        images: List[str],
        model: str = "openai/clip-vit-base-patch32",
        timeout: Optional[int] = None
    ) -> List[List[float]]:
        """
        Get image features using CLIP model.

        Args:
            images: List of image paths or URLs
            model: CLIP model name
            timeout: Request timeout in seconds

        Returns:
            List of image feature vectors
        """
        payload: Dict[str, Any] = {
            "features": "image",
            "images": images
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="clip",
            timeout=timeout
        )

        return response["response"]

    def get_text_features_HF(
        self,
        texts: List[str],
        model: str = "openai/clip-vit-base-patch32",
        timeout: Optional[int] = None
    ) -> List[List[float]]:
        """
        Get text features using CLIP model.

        Args:
            texts: List of text strings
            model: CLIP model name
            timeout: Request timeout in seconds

        Returns:
            List of text feature vectors
        """
        payload: Dict[str, Any] = {
            "features": "text",
            "text": texts
        }

        response = self._make_request(
            payload=payload,
            model_name=model,
            model_type="clip",
            timeout=timeout
        )

        return response["response"]

    # ========== Utility Methods ==========

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: str, exc_val: str, exc_tb: str):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Create client
    client = ModelServerClient(base_url="http://localhost:1121")

    try:
        # Example 1: Create embeddings
        print("=" * 50)
        print("Example 1: Creating embeddings")
        print("=" * 50)
        texts = ["Hello world", "How are you?", "Machine learning is amazing"]
        embeddings = client.create_openai_embeddings(texts)
        print(f"Created {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}")

        # Example 2: Chat completion
        print("\n" + "=" * 50)
        print("Example 2: Chat completion with OpenAI")
        print("=" * 50)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = client.openai_chat_completion(messages)
        print(f"Response: {response}")

        # Example 3: Sentence transformers
        print("\n" + "=" * 50)
        print("Example 3: Sentence transformers")
        print("=" * 50)
        sentences = ["This is a test sentence", "Another test sentence"]
        sent_embeddings = client.create_transformer_embeddings(
            sentences,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"Created {len(sent_embeddings)} sentence embeddings")
        print(f"Embedding dimension: {len(sent_embeddings[0])}")

        # Example 4: Claude completion
        print("\n" + "=" * 50)
        print("Example 4: Claude completion")
        print("=" * 50)
        claude_messages = [
            {"role": "user", "content": "Explain what a neural network is in one sentence."}
        ]
        claude_response = client.claude_chat_completion(claude_messages)
        print(f"Claude response: {claude_response}")

    except Exception as e:
        logger.error(f"Error during example execution: {str(e)}")
    finally:
        client.close()
