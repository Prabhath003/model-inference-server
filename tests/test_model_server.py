import pytest
from src.model_server import ModelServer


def test_model_server_initialization():
    server = ModelServer(port=4999, model_name="gpt-4o-mini", type="text-generation")
    assert server.model_name == "gpt-4o-mini"
    assert server.type == "text-generation"
    assert server.port == 4999


def test_model_server_invalid_type():
    with pytest.raises(ValueError):
        ModelServer(port=4999, model_name="gpt-4o-mini", type="invalid-type")


def test_start_server_no_available_gpus(mocker):
    mocker.patch("src.utils.get_optimal_gpu_set", return_value=[])
    server = ModelServer(port=4999, model_name="gpt-4o-mini", type="text-generation")
    with pytest.raises(RuntimeError):
        server.start_server()


def test_start_server_success(mocker):
    mocker.patch("src.utils.get_optimal_gpu_set", return_value=[0])
    server = ModelServer(port=4999, model_name="gpt-4o-mini", type="text-generation")
    # Mock Flask app run method
    mock_app = mocker.patch("src.model_server.Flask.run")
    server.start_server()
    mock_app.assert_called_once_with(port=4999)
