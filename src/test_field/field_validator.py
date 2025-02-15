import pytest
from core.field import GridInfo


class FieldValidator:
    def __init__(self, shape, rtol=1e-5, atol=1e-8):
        self.grid_info = GridInfo(shape=shape, dx=tuple(1.0 / (s - 1) for s in shape))
        self.rtol = rtol
        self.atol = atol

    def run_validation_suite(self):
        test_files = [
            "test_scalar_field.py",
            "test_vector_field.py",
            "test_field_operations.py",
        ]

        results = []
        for test_file in test_files:
            result = pytest.main([f"test_field/{test_file}", "-v"])
            results.append(result == pytest.ExitCode.OK)

        return all(results)
