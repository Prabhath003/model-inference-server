import unittest
from src.manager import monitor_processes, processes


class TestManager(unittest.TestCase):

    def setUp(self):
        # Set up any necessary state before each test
        processes.clear()  # Clear any existing processes for a clean test environment

    def test_process_management(self):
        # Simulate adding a process
        processes["test_model"] = {"last_response_time": 0, "pid": 1234, "port": 5000}

        # Call the monitor_processes function
        monitor_processes()

        # Check if the process was removed due to inactivity
        self.assertNotIn("test_model", processes)

    def test_process_inactivity(self):
        # Simulate adding a process with a last_response_time set to 11 minutes ago
        processes["inactive_model"] = {
            "last_response_time": time.time() - 660,
            "pid": 1234,
            "port": 5001,
        }

        # Call the monitor_processes function
        monitor_processes()

        # Check if the process was removed due to inactivity
        self.assertNotIn("inactive_model", processes)


if __name__ == "__main__":
    unittest.main()
