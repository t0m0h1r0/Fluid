import jax.numpy as jnp
import numpy as np
from numerics.poisson import PoissonSORSolver, PoissonMultigridSolver, PoissonCGSolver
from numerics.poisson import PoissonConfig
from core.field import ScalarField, GridInfo

# 解析的に解ける関数を設定
def analytical_solution(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

# ラプラシアンを解析的に計算
def analytical_laplacian(x, y, z):
    return -3 * (np.pi ** 2) * analytical_solution(x, y, z)

# テストケースの設定
def test_poisson_solvers():
    # グリッドの設定
    nx, ny, nz = 64, 64, 64
    shape = (nx, ny, nz)
    dx = (1.0 / (nx - 1), 1.0 / (ny - 1), 1.0 / (nz - 1))
    grid = GridInfo(shape=shape, dx=dx)

    # 座標の生成
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 解析解とラプラシアンの計算（NumPyで計算）
    analytical_sol = analytical_solution(X, Y, Z)
    analytical_lap = analytical_laplacian(X, Y, Z)

    # NumPy配列をJAX配列に変換
    analytical_lap_jax = jnp.array(analytical_lap)

    # ソルバーの設定
    config = PoissonConfig()
    solvers = [
        PoissonSORSolver(config),
        PoissonMultigridSolver(config),
        PoissonCGSolver(config),
    ]

    # 各ソルバーをテスト
    for solver in solvers:
        print(f"Testing {solver.__class__.__name__}")
        # JAX配列を直接ソルバーに渡す
        numerical_sol = solver.solve(analytical_lap_jax)
        
        # 結果をNumPy配列に変換して比較
        numerical_array = np.array(numerical_sol)
        error = np.max(np.abs(numerical_array - analytical_sol))
        print(f"Max error: {error:.6e}")
        assert error < 1e-6, f"Error is too large for {solver.__class__.__name__}"

if __name__ == "__main__":
    test_poisson_solvers()