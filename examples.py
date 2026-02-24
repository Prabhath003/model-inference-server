"""
Example usage scripts for the Model Server API Client.

This module demonstrates various use cases for the model server.
"""

from api_client import ModelServerClient
import numpy as np


def example_embeddings():
    """Example: Create embeddings and compute similarity."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Creating Embeddings and Computing Similarity")
    print("=" * 70)

    client = ModelServerClient()

    # Create embeddings for similar sentences
    sentences = [
        "The cat sits on the mat",
        "A feline rests on the rug",
        "Python is a programming language",
        "Java is used for software development",
    ]

    print(f"\nInput sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")

    # Get embeddings
    embeddings = client.create_openai_embeddings(sentences)

    print(f"\nCreated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Compute cosine similarity
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    print("\nSimilarity matrix:")
    print("     ", "  ".join([f"S{i+1}" for i in range(len(sentences))]))
    for i, emb1 in enumerate(embeddings):
        similarities = []
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1, emb2)
            similarities.append(f"{sim:.3f}")
        print(f"S{i+1}:  " + "  ".join(similarities))

    client.close()


def example_chat_completion():
    """Example: Multi-turn conversation with OpenAI."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Multi-turn Conversation with OpenAI")
    print("=" * 70)

    client = ModelServerClient()

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "How do I read a CSV file in Python?"},
    ]

    print("\nUser: How do I read a CSV file in Python?")

    response = client.openai_chat_completion(messages, temperature=0.7)
    print(f"\nAssistant: {response}")

    # Follow-up question
    messages.append({"role": "assistant", "content": response})
    messages.append(
        {"role": "user", "content": "Can you show me an example with pandas?"}
    )

    print("\nUser: Can you show me an example with pandas?")

    response = client.claude_chat_completion(messages, temperature=0.7)
    print(f"\nAssistant: {response}")

    client.close()


def example_semantic_search():
    """Example: Semantic search using embeddings."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Semantic Search")
    print("=" * 70)

    client = ModelServerClient()

    # Document corpus
    documents = [
        "Python is a high-level programming language",
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "JavaScript is commonly used for web development",
        "Deep learning uses multiple layers of neural networks",
        "React is a JavaScript library for building user interfaces",
    ]

    # Query
    query = "Tell me about AI and neural networks"

    print(f"\nDocument corpus ({len(documents)} documents):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    print(f"\nQuery: {query}")

    # Create embeddings
    all_texts = documents + [query]
    embeddings = client.create_transformer_embeddings(all_texts)

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute similarities
    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim, documents[i]))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 most relevant documents:")
    for rank, (idx, sim, doc) in enumerate(similarities[:3], 1):
        print(f"  {rank}. [Score: {sim:.4f}] {doc}")

    client.close()


def example_batch_processing():
    """Example: Batch processing with embeddings."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Batch Processing")
    print("=" * 70)

    client = ModelServerClient()

    # Large batch of texts
    texts = [f"This is document number {i} about various topics" for i in range(1, 11)]

    print(f"\nProcessing {len(texts)} documents in a single batch...")

    # Process all at once
    embeddings = client.create_transformer_embeddings(texts)

    print(f"Successfully created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"\nFirst document embedding (first 10 values):")
    print(f"  {embeddings[0][:10]}")

    client.close()


def example_rag_pipeline():
    """Example: Simple RAG (Retrieval-Augmented Generation) pipeline."""
    print("\n" + "=" * 70)
    print("EXAMPLE: RAG Pipeline")
    print("=" * 70)

    client = ModelServerClient()

    # Knowledge base
    knowledge_base = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "The Amazon rainforest is the largest tropical rainforest in the world.",
        "Albert Einstein developed the theory of relativity in the early 20th century.",
        "The Great Wall of China is over 13,000 miles long.",
    ]

    query = "When was Python created?"

    print(f"\nKnowledge Base ({len(knowledge_base)} documents):")
    for i, doc in enumerate(knowledge_base, 1):
        print(f"  {i}. {doc}")

    print(f"\nQuery: {query}")

    # Step 1: Retrieve relevant documents
    print("\nStep 1: Retrieving relevant documents...")
    all_texts = knowledge_base + [query]
    embeddings = client.create_openai_embeddings(all_texts)

    kb_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = []
    for i, doc_emb in enumerate(kb_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, knowledge_base[i]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in similarities[:2]]

    print(f"Retrieved top {len(top_docs)} documents:")
    for i, doc in enumerate(top_docs, 1):
        print(f"  {i}. {doc}")

    # Step 2: Generate answer using retrieved context
    print("\nStep 2: Generating answer with context...")
    context = "\n".join(top_docs)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based on the provided context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
        },
    ]

    answer = client.openai_chat_completion(messages, temperature=0.3)
    print(f"\nAnswer: {answer}")

    client.close()


def example_sentiment_analysis():
    """Example: Sentiment analysis using embeddings."""
    print("\n" + "=" * 70)
    print("EXAMPLE: Sentiment Analysis using Embeddings")
    print("=" * 70)

    client = ModelServerClient()

    # Reference sentiment examples
    positive_ref = "This is absolutely wonderful and amazing!"
    negative_ref = "This is terrible and awful!"

    # Test sentences
    test_sentences = [
        "I love this product, it's fantastic!",
        "This is the worst experience ever.",
        "It's okay, nothing special.",
        "Absolutely brilliant and outstanding!",
        "Very disappointed and unhappy.",
    ]

    print("\nReference sentences:")
    print(f"  Positive: {positive_ref}")
    print(f"  Negative: {negative_ref}")

    print("\nTest sentences:")
    for i, sent in enumerate(test_sentences, 1):
        print(f"  {i}. {sent}")

    # Get embeddings
    all_texts = [positive_ref, negative_ref] + test_sentences
    embeddings = client.create_openai_embeddings(all_texts)

    pos_emb = embeddings[0]
    neg_emb = embeddings[1]
    test_embs = embeddings[2:]

    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    print("\nSentiment Analysis Results:")
    for i, (sent, test_emb) in enumerate(zip(test_sentences, test_embs), 1):
        pos_sim = cosine_similarity(test_emb, pos_emb)
        neg_sim = cosine_similarity(test_emb, neg_emb)

        sentiment = "POSITIVE" if pos_sim > neg_sim else "NEGATIVE"
        confidence = abs(pos_sim - neg_sim)

        print(f'\n  {i}. "{sent}"')
        print(f"     Sentiment: {sentiment} (confidence: {confidence:.4f})")
        print(f"     Positive similarity: {pos_sim:.4f}")
        print(f"     Negative similarity: {neg_sim:.4f}")

    client.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" MODEL SERVER API CLIENT - EXAMPLES")
    print("=" * 70)

    examples = [
        ("Embeddings and Similarity", example_embeddings),
        ("Chat Completion", example_chat_completion),
        ("Semantic Search", example_semantic_search),
        ("Batch Processing", example_batch_processing),
        ("RAG Pipeline", example_rag_pipeline),
        ("Sentiment Analysis", example_sentiment_analysis),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {str(e)}")

    print("\n" + "=" * 70)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
