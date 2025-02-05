# numerics/poisson_solver/classic_poisson_solver.py
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from core.scheme import DifferenceScheme


class ClassicPoissonSolver:
    def __init__(self, scheme: DifferenceScheme, boundary_conditions: list):
        """
        クラシック（直接法）ポアソンソルバーの初期化

        Args:
            scheme: 差分スキーム
            boundary_conditions: 境界条件のリスト（x, y, z順）
        """
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions

    def solve(
        self, rhs: np.ndarray, tolerance: float = 1e-6, max_iterations: int = 100
    ) -> np.ndarray:
        """
        ポアソン方程式を解く（直接法）

        Args:
            rhs: 右辺項（ソース項）
            tolerance: 収束判定の許容誤差（この引数は直接法では使用されないが、互換性のために保持）
            max_iterations: 最大反復回数（直接法では使用されないが、互換性のために保持）

        Returns:
            圧力場
        """
        # 形状と次元の確認
        shape = rhs.shape
        ndim = rhs.ndim

        # 3次元配列を想定
        if ndim != 3:
            raise ValueError(f"3次元配列が必要です。現在の配列次元: {ndim}")

        Nx, Ny, Nz = shape

        # 係数行列の構築
        A = self._build_matrix(shape)

        # 右辺項をベクトル化
        b = rhs.flatten()

        # 直接法で解く
        try:
            # スパース行列を使用した直接解法
            x = spsolve(A, b)

            # 圧力場を元の形状に戻す
            pressure = x.reshape(shape)

            return pressure

        except Exception as e:
            print(f"ポアソン方程式の求解中にエラーが発生: {e}")
            raise

    def _build_matrix(self, shape: tuple) -> csr_matrix:
        """
        3次元ポアソン方程式の係数行列を構築

        Args:
            shape: グリッドの形状 (Nx, Ny, Nz)

        Returns:
            スパース行列（CSR形式）
        """
        Nx, Ny, Nz = shape
        N = Nx * Ny * Nz

        # 対角成分の初期化
        main_diag = np.zeros(N)
        lower_diag = np.zeros(N - 1)
        upper_diag = np.zeros(N - 1)

        # 隣接グリッド間の係数（等間隔グリッドを仮定）
        dx = dy = dz = 1.0

        # 対角成分の計算
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    idx = i + j * Nx + k * Nx * Ny

                    # 対角成分
                    main_diag[idx] = -(2 / (dx * dx) + 2 / (dy * dy) + 2 / (dz * dz))

                    # x方向の隣接項
                    if i > 0:
                        lower_diag[idx - 1] = 1 / (dx * dx)
                    if i < Nx - 1:
                        upper_diag[idx] = 1 / (dx * dx)

        # スパース行列の構築
        diagonals = [main_diag, lower_diag, upper_diag]
        offsets = [0, -1, 1]

        A = diags(diagonals, offsets, shape=(N, N), format="csr")

        return A
