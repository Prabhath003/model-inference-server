import requests


def test_openai_inference():
    response = requests.post(
        "http://localhost:5000/infer",
        json={
            "type": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "your_api_key_here",
            "input_text": [
                {"role": "system", "content": "You are the Iron Man!"},
                {"role": "user", "content": "Tell me how Jarvis works!"},
            ],
            "temperature": 0.9,
            "max_new_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "choices" in response.json()  # Adjust based on actual response structure


if __name__ == "__main__":
    test_openai_inference()
