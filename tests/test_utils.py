import unittest
from src.utils import (
    get_model_size,
    estimate_memory,
    get_available_gpus,
    select_optimal_gpus,
)


class TestUtils(unittest.TestCase):

    def test_get_model_size(self):
        model_name = "bert-base-uncased"
        size = get_model_size(model_name)
        self.assertIsNotNone(size)
        self.assertGreater(size, 0)

    def test_estimate_memory(self):
        model_name = "bert-base-uncased"
        model_size, activation_memory, total_memory = estimate_memory(model_name)
        self.assertIsNotNone(model_size)
        self.assertIsNotNone(activation_memory)
        self.assertIsNotNone(total_memory)
        self.assertGreater(total_memory, 0)

    def test_get_available_gpus(self):
        gpus = get_available_gpus()
        self.assertIsInstance(gpus, list)

    def test_select_optimal_gpus(self):
        required_memory = 8.0  # GB
        selected_gpus = select_optimal_gpus(required_memory)
        self.assertIsInstance(selected_gpus, list)


if __name__ == "__main__":
    unittest.main()
