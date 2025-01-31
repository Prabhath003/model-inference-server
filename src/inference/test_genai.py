import requests

def test_genai_inference():
    response = requests.post("http://localhost:5000/infer", json={
        "type": "gen-ai",
        "model_name": "gemini-1.5-flash",
        "api_key": "YOUR_API_KEY_HERE",
        "input_text": [
            {"role": "system", "content": "You are the Iron Man!"},
            {"role": "user", "content": "Tell me how Jarvis works!"}
        ],
        "temperature": 0.9,
        "max_new_tokens": 1024
    })
    
    assert response.status_code == 200
    assert "output" in response.json()  # Adjust based on actual response structure

if __name__ == "__main__":
    test_genai_inference()