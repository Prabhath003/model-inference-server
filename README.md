# Model Inference Server

This project is a Python-based server for performing model inference using various machine learning models. It provides a flexible architecture to serve different types of models, including text generation, OpenAI models, GenAI models, and sentence transformation models.

## Project Structure

```
model-inference-server
├── src
│   ├── __init__.py
│   ├── app.py
│   ├── manager.py
│   ├── model_server.py
│   ├── utils.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── test_openai.py
│   │   ├── test_genai.py
│   │   ├── test_text_generation.py
│   │   └── test_sent_trans.py
│   └── routes
│       ├── __init__.py
│       └── infer.py
├── tests
│   ├── __init__.py
│   ├── test_manager.py
│   ├── test_model_server.py
│   └── test_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd model-inference-server
pip install -r requirements.txt
```

## Usage

To start the server, run the following command:

```bash
python -m src.app
```

The server will start and listen for inference requests on the specified port.

## Testing

To run the tests, use the following command:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.