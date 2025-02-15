import numpy as np
import pytest
from core.field import ScalarField


@pytest.fixture
def sample_scalar_field(field_validator):
    x, y, z = np.meshgrid(
        *[np.linspace(0, 1, s) for s in field_validator.grid_info.shape], indexing="ij"
    )
    data = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(z)
    return ScalarField(field_validator.grid_info, initial_value=data)


def test_scalar_field_gradient(field_validator, sample_scalar_field):
    field = sample_scalar_field
    gradient = field.gradient()

    x, y, z = np.meshgrid(
        *[np.linspace(0, 1, s) for s in field_validator.grid_info.shape], indexing="ij"
    )
    expected_gradient_x = (
        2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(z)
    )
    expected_gradient_y = (
        -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(z)
    )
    expected_gradient_z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.exp(z)

    for i, (comp, expected) in enumerate(
        zip(
            gradient.components,
            [expected_gradient_x, expected_gradient_y, expected_gradient_z],
        )
    ):
        assert np.allclose(
            comp.data, expected, rtol=field_validator.rtol, atol=field_validator.atol
        )


# ... 他のテストケースを追加 ...
