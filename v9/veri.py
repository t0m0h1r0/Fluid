import numpy as np
import matplotlib.pyplot as plt
from physics.poisson.sor import SORSolver


class Poisson3DVerification:
    """3次元ポアソン方程式の検証クラス"""

    def __init__(self):
        """初期化時にイテレーション回数を追跡するための変数"""
        self.iteration_count = 0

    @staticmethod
    def exact_solution(x, y, z):
        """
        3D解析解
        u(x,y,z) = x^2 * y^2 * z^2 + x * y * z + sin(π*x) * sin(π*y) * sin(π*z)

        この関数は以下の性質を持つ：
        1. 境界条件を満たす
        2. 非自明な勾配と曲率を持つ
        3. 多項式と三角関数の組み合わせで複雑性を表現
        """
        return (
            x**2 * y**2 * z**2  # 2次多項式成分
            +
            # x * y * z +             # 線形混合項
            # np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)  # 三角関数成分
            0
        )

    @staticmethod
    def source_term(x, y, z):
        """
        対応する右辺（ソース項）
        ∇²u = f を満たすf（ラプラシアンを計算）
        """
        # 多項式項のラプラシアン
        laplacian_poly = 2 * (y**2 * z**2 + x**2 * z**2 + x**2 * y**2)

        # 線形項のラプラシアン
        laplacian_linear = 0

        # 三角関数項のラプラシアン
        laplacian_trig = (
            -(3 * np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        )

        return laplacian_poly + laplacian_linear + laplacian_trig

    def verify_poisson_3d(
        self, nx=32, ny=32, nz=32, omega=1.5, tolerance=1e-6, max_iterations=1000
    ):
        """3Dポアソン方程式の検証"""
        # 領域の設定
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # グリッド間隔
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        # 厳密解と右辺の計算
        u_exact = self.exact_solution(X, Y, Z)
        rhs = self.source_term(X, Y, Z)

        # SORソルバーの設定
        solver = SORSolver(
            omega=omega, tolerance=tolerance, max_iterations=max_iterations
        )

        # 初期解
        u_init = np.zeros_like(rhs)

        # 求解 - 重要：グリッド間隔を配列で渡す
        result = solver.solve(
            initial_solution=u_init,
            rhs=rhs,
            dx=[dx, dy, dz],  # リストとして渡すことで多次元対応
        )

        # イテレーション回数の記録
        self.iteration_count = solver.iteration_count

        # 誤差評価
        error = np.abs(result - u_exact)
        l2_error = np.sqrt(np.mean(error**2))
        l_inf_error = np.max(np.abs(error))
        relative_error = l2_error / np.max(np.abs(u_exact))

        # 結果の表示
        print("3D Poisson方程式検証結果:")
        print(f"グリッドサイズ: {nx} x {ny} x {nz}")
        print(f"グリッド間隔: dx={dx}, dy={dy}, dz={dz}")
        print(f"緩和係数 (ω): {omega}")
        print(f"反復回数: {solver.iteration_count}")
        print(f"L2誤差: {l2_error}")
        print(f"最大絶対誤差: {l_inf_error}")
        print(f"相対誤差: {relative_error}")
        print(f"収束履歴の長さ: {len(solver.residual_history)}")

        # 誤差の可視化
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title("厳密解（中央断面）")
        plt.imshow(u_exact[:, :, nz // 2], cmap="viridis")
        plt.colorbar()

        plt.subplot(132)
        plt.title("数値解（中央断面）")
        plt.imshow(result[:, :, nz // 2], cmap="viridis")
        plt.colorbar()

        plt.subplot(133)
        plt.title("絶対誤差（中央断面）")
        plt.imshow(error[:, :, nz // 2], cmap="hot")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        return result, u_exact, error

    def convergence_study(self):
        """グリッドサイズに対する収束性の研究"""
        grid_sizes = [16, 32, 64, 128]
        errors = []
        iterations = []

        plt.figure(figsize=(15, 5))

        for nx in grid_sizes:
            ny = nz = nx
            print(f"\nグリッドサイズ: {nx} x {ny} x {nz}")

            # 解の検証
            _, _, error = self.verify_poisson_3d(nx, ny, nz)

            # 誤差の計算
            l2_error = np.sqrt(np.mean(error**2))
            errors.append(l2_error)
            iterations.append(self.iteration_count)

        # 収束性解析のプロット
        plt.subplot(121)
        plt.title("グリッドサイズ vs L2誤差")
        plt.loglog(grid_sizes, errors, marker="o")
        plt.xlabel("グリッドサイズ")
        plt.ylabel("L2誤差")
        plt.grid(True)

        plt.subplot(122)
        plt.title("グリッドサイズ vs 反復回数")
        plt.plot(grid_sizes, iterations, marker="o")
        plt.xlabel("グリッドサイズ")
        plt.ylabel("反復回数")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# メイン実行
if __name__ == "__main__":
    verifier = Poisson3DVerification()

    # 3D Poissonの検証
    print("3D Poisson方程式の詳細検証:")
    verifier.verify_poisson_3d()

    # 収束性研究
    print("\n収束性研究:")
    verifier.convergence_study()
