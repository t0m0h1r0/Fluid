import numpy as np
import pytest
from core.field import ScalarField


@pytest.fixture
def sample_scalar_field(field_validator):
    return ScalarField(field_validator.grid_info, initial_value=2.0)


def test_scalar_field_initialization(field_validator):
    field = ScalarField(field_validator.grid_info, initial_value=1.0)
    assert field.shape == field_validator.grid_info.shape
    assert field.dx == field_validator.grid_info.dx
    assert np.allclose(
        field.data, 1.0, rtol=field_validator.rtol, atol=field_validator.atol
    )


def test_scalar_field_addition(field_validator, sample_scalar_field):
    field = sample_scalar_field
    result = field + 2.0
    assert np.allclose(
        result.data, 4.0, rtol=field_validator.rtol, atol=field_validator.atol
    )

    other_field = ScalarField(field.grid, initial_value=3.0)
    result = field + other_field
    assert np.allclose(
        result.data, 5.0, rtol=field_validator.rtol, atol=field_validator.atol
    )


# ... 他のテストケースを追加 ...
