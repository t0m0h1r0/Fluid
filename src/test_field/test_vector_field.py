import numpy as np
import pytest
from core.field import VectorField


@pytest.fixture
def sample_vector_field(field_validator):
    return VectorField(field_validator.grid_info, initial_values=(1.0, 2.0, 3.0))


def test_vector_field_initialization(field_validator):
    field = VectorField(field_validator.grid_info, initial_values=(1.0, 2.0, 3.0))
    assert field.shape == field_validator.grid_info.shape
    assert field.dx == field_validator.grid_info.dx
    for i, comp in enumerate(field.components):
        assert np.allclose(
            comp.data, i + 1, rtol=field_validator.rtol, atol=field_validator.atol
        )


def test_vector_field_addition(field_validator, sample_vector_field):
    field = sample_vector_field
    result = field + VectorField(field.grid, initial_values=(1.0, 1.0, 1.0))
    for i, comp in enumerate(result.components):
        assert np.allclose(
            comp.data, i + 2, rtol=field_validator.rtol, atol=field_validator.atol
        )


# ... 他のテストケースを追加 ...
