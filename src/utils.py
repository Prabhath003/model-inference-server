from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoConfig
import torch
import pynvml
import socket
from typing import Literal, List, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_size(model_name: str) -> float:
    """
    Fetch the model size (in GB) from the Hugging Face safetensors index.

    Args:
        model_name (str): The name of the model.

    Returns:
        float: The size of the model in GB, or None if an error occurs.
    """
    try:
        repo_files = list_repo_files(model_name)
        safetensors_files = [
            file for file in repo_files if file.endswith(".safetensors")
        ]

        total_size_bytes = 0
        for file in safetensors_files:
            file_path = hf_hub_download(model_name, file)
            total_size_bytes += os.path.getsize(file_path)

        total_size_gb = total_size_bytes / 1e9  # Convert bytes to GB
        logger.info(f"Model size for {model_name}: {total_size_gb} GB")
        return total_size_gb

    except Exception as e:
        logger.error(f"Error fetching model size: {e}")
        return None


def estimate_memory(
    model_name: str,
    dtype: Literal["fp32", "fp16", "int8"] = "fp16",
    batch_size: int = 8,
    seq_length: int = 8192,
) -> Tuple[float, float, float]:
    """
    Estimate total memory required for model inference.
    - Uses precomputed model size (weights).
    - Estimates activation memory based on batch size & sequence length.

    Args:
        model_name (str): The name of the model.
        dtype (Literal["fp32", "fp16", "int8"], optional): The data type. Defaults to "fp16".
        batch_size (int, optional): The batch size. Defaults to 8.
        seq_length (int, optional): The sequence length. Defaults to 8192.

    Returns:
        Tuple[float, float, float]: The model size in GB, activation memory in GB, and total memory in GB.
    """
    model_size_gb = get_model_size(model_name)

    if model_size_gb is None:
        logger.warning("Could not retrieve model size, using fallback method.")
        try:
            config = AutoConfig.from_pretrained(model_name)
            num_params = getattr(
                config, "num_parameters", lambda: 0
            )()  # Safely get num_parameters
            dtype_size = {"fp32": 4, "fp16": 2, "int8": 1}.get(dtype, 2)
            model_size_gb = (num_params * dtype_size) / 1e9  # Convert bytes to GB
        except Exception as e:
            logger.error(f"Error estimating model size from config: {e}")
            return None, None, None

    dtype_offset = {"fp32": 4, "fp16": 2, "int8": 1}.get(dtype, 2)
    activation_memory: float = model_size_gb * 0.2 * dtype_offset

    total_memory: float = model_size_gb * 1.2
    logger.info(
        f"Estimated memory for {model_name} with dtype {dtype}: model_size_gb={model_size_gb}, activation_memory={activation_memory}, total_memory={total_memory}"
    )
    return model_size_gb, activation_memory, total_memory


def get_available_gpus() -> List[Tuple[int, float]]:
    """
    Get a list of available GPUs and their free memory.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing GPU index and free memory in GB.
    """
    try:
        pynvml.nvmlInit()
        gpu_info: List[Tuple[int, float]] = []
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            free_mem: float = (
                pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1e9
            )  # Convert to GB
            gpu_info.append((i, free_mem))
        pynvml.nvmlShutdown()
        logger.info(f"Available GPUs: {gpu_info}")
        return sorted(gpu_info, key=lambda x: x[1], reverse=True)  # Sort by free memory
    except pynvml.NVMLError as e:
        logger.error(f"Error initializing NVML: {e}")
        return []


def select_optimal_gpus(required_memory: float) -> List[int]:
    """
    Select the optimal GPUs based on the required memory.

    Args:
        required_memory (float): The required memory in GB.

    Returns:
        List[int]: A list of selected GPU indices.
    """
    gpus = get_available_gpus()

    possible_gpus: List[Tuple[int, float]] = []
    possible_gpus.extend(
        (gpu, free_mem) for gpu, free_mem in gpus if free_mem >= required_memory
    )
    if possible_gpus:
        possible_gpus = sorted(possible_gpus, key=lambda x: x[1], reverse=False)
        logger.info(
            f"Selected GPU: {possible_gpus[0][0]} with {possible_gpus[0][1]} GB free memory"
        )
        return [possible_gpus[0][0]]
    else:
        selected_gpus: List[int] = []
        total_available: float = 0
        gpus = sorted(gpus, key=lambda x: x[1], reverse=True)
        for gpu, free_mem in gpus:
            selected_gpus.append(gpu)
            total_available += free_mem
            if total_available >= required_memory:
                logger.info(
                    f"Selected GPUs: {selected_gpus} with total available memory: {total_available} GB"
                )
                return selected_gpus  # Use multiple GPUs

    logger.warning("Not enough memory available on any GPU")
    return []  # Not enough memory available


def get_optimal_gpu_set(
    model_name: str, dtype: Literal["fp32", "fp16", "int8"] = "fp16"
) -> List[int]:
    """
    Get the optimal set of GPUs for the given model and data type.

    Args:
        model_name (str): The name of the model.
        dtype (Literal["fp32", "fp16", "int8"], optional): The data type. Defaults to "fp16".

    Returns:
        List[int]: A list of selected GPU indices.
    """
    _, _, total_mem = estimate_memory(model_name, dtype)
    if total_mem is None:
        logger.error("Error estimating memory requirements.")
        return []
    return select_optimal_gpus(total_mem)


def get_available_ports(start: int = 5001, end: int = 5100):
    logging.info("Gathering available ports...")
    available_ports: list[int] = []
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", port)) != 0:  # Port is available
                available_ports.append(port)
    logging.info(f"Available ports: {available_ports}")
    return available_ports
