import requests

def test_sent_trans():
    response = requests.post("http://localhost:5000/infer", json={
        "type": "sent-trans",
        "model_name": "jinaai/jina-embeddings-v2-base-en",
        "input_text": [
            "You are the Iron Man!",
            "Tell me how Jarvis works!"
        ]
    })
    assert response.status_code == 200
    assert "embedding" in response.json()  # Adjust based on expected response structure

if __name__ == "__main__":
    test_sent_trans()