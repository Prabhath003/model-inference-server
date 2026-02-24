import requests


def test_text_generation_inference():
    response = requests.post(
        "http://localhost:5000/infer",
        json={
            "type": "text-generation",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "input_text": [
                {"role": "system", "content": "You are the Iron Man!"},
                {"role": "user", "content": "Tell me how Jarvis works!"},
            ],
            "temperature": 0.9,
            "max_new_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "output" in response.json()  # Adjust based on actual response structure


if __name__ == "__main__":
    test_text_generation_inference()
