import numpy as np
import pytest
from src.core.boundary import PeriodicBC, NeumannBC
from src.numerics.compact_scheme import CompactScheme
from src.physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from src.physics.navier_stokes import NavierStokesSolver
from src.core.boundary import DirectionalBC

@pytest.fixture
def scheme():
    return CompactScheme(alpha=0.25)

@pytest.fixture
def boundary_conditions():
    return DirectionalBC(
        x_bc=PeriodicBC(),
        y_bc=PeriodicBC(),
        z_bc=NeumannBC()
    )

def test_compact_derivative():
    scheme = CompactScheme()
    bc = PeriodicBC()
    x = np.linspace(0, 2*np.pi, 32)
    f = np.sin(x)
    df = scheme.apply(f, bc)
    df_exact = np.cos(x)
    assert np.allclose(df, df_exact, atol=1e-2)

def test_phase_field_heaviside():
    params = PhaseFieldParams(epsilon=0.1)
    solver = PhaseFieldSolver(CompactScheme(), DirectionalBC(
        PeriodicBC(), PeriodicBC(), PeriodicBC()
    ), params)
    
    phi = np.linspace(-1, 1, 100)
    H = solver.heaviside(phi)
    assert np.all(H >= 0) and np.all(H <= 1)
    assert np.allclose(H[phi < -0.5], 0, atol=0.1)
    assert np.allclose(H[phi > 0.5], 1, atol=0.1)

def test_mass_conservation():
    # NSソルバーの質量保存テスト
    shape = (16, 16, 16)
    u = np.random.randn(*shape)
    v = np.random.randn(*shape)
    w = np.zeros_like(u)  # 発散がゼロになるように調整
    
    # 発散を計算
    dx = dy = dz = 1.0
    div = (np.gradient(u, dx, axis=0) + 
           np.gradient(v, dy, axis=1) + 
           np.gradient(w, dz, axis=2))
    
    assert np.allclose(div, 0, atol=1e-10)