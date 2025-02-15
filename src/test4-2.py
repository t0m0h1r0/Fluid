"""
ScalarFieldの微分演算をテストするモジュール

以下の演算を検証:
- 勾配 (gradient)
- 発散 (divergence)
- ラプラシアン
- 各方向の偏微分
"""

import unittest
import jax.numpy as jnp
from pathlib import Path
import sys

# core/fieldへのパスを追加
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.field import ScalarField, VectorField, GridInfo


class TestScalarFieldDerivatives(unittest.TestCase):
    """ScalarFieldの微分演算テスト"""

    def setUp(self):
        """テストの前準備"""
        # 3D格子の設定
        self.nx, self.ny, self.nz = 16, 16, 16
        self.Lx, self.Ly, self.Lz = 2.0, 2.0, 2.0
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.dz = self.Lz / (self.nz - 1)

        self.shape = (self.nx, self.ny, self.nz)
        self.grid = GridInfo(shape=self.shape, dx=(self.dx, self.dy, self.dz))

        # 座標格子の生成
        x = jnp.linspace(0, self.Lx, self.nx)
        y = jnp.linspace(0, self.Ly, self.ny)
        z = jnp.linspace(0, self.Lz, self.nz)
        self.X, self.Y, self.Z = jnp.meshgrid(x, y, z, indexing="ij")

        # テスト関数の設定（解析的に微分可能な関数）
        self.data = (
            jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )
        self.field = ScalarField(self.grid, self.data)

    def test_gradient_individual_components(self):
        """各方向の偏微分のテスト"""
        # X方向の解析解
        expected_dx = (
            (2 * jnp.pi / self.Lx)
            * jnp.cos(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )
        grad_x = self.field.gradient(axis=0)
        jnp.testing.assert_allclose(grad_x.data, expected_dx, rtol=1e-4)

        # Y方向の解析解
        expected_dy = (
            (-2 * jnp.pi / self.Ly)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.sin(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )
        grad_y = self.field.gradient(axis=1)
        jnp.testing.assert_allclose(grad_y.data, expected_dy, rtol=1e-4)

        # Z方向の解析解
        expected_dz = (
            (-1 / self.Lz)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )
        grad_z = self.field.gradient(axis=2)
        jnp.testing.assert_allclose(grad_z.data, expected_dz, rtol=1e-4)

    def test_gradient_full(self):
        """全勾配ベクトルのテスト"""
        grad = self.field.gradient()
        self.assertIsInstance(grad, VectorField)
        self.assertEqual(len(grad.components), 3)

        # 各成分の検証
        expected_grads = [
            # X方向
            (2 * jnp.pi / self.Lx)
            * jnp.cos(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz),
            # Y方向
            (-2 * jnp.pi / self.Ly)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.sin(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz),
            # Z方向
            (-1 / self.Lz)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz),
        ]

        for i, expected in enumerate(expected_grads):
            jnp.testing.assert_allclose(grad.components[i].data, expected, rtol=1e-4)

    def test_divergence(self):
        """発散（ラプラシアン）のテスト"""
        # ラプラシアンの解析解
        expected_laplacian = (
            # ∂²/∂x²項
            (-4 * jnp.pi**2 / self.Lx**2)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
            +
            # ∂²/∂y²項
            (-4 * jnp.pi**2 / self.Ly**2)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
            +
            # ∂²/∂z²項
            (1 / self.Lz**2)
            * jnp.sin(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )

        laplacian = self.field.divergence()
        jnp.testing.assert_allclose(laplacian.data, expected_laplacian, rtol=1e-4)

    def test_mixed_derivatives(self):
        """混合偏微分のテスト（∂²/∂x∂y など）"""
        # 解析解: ∂²f/∂x∂y
        expected_dxdy = (
            (-4 * jnp.pi**2 / (self.Lx * self.Ly))
            * jnp.cos(2 * jnp.pi * self.X / self.Lx)
            * jnp.sin(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )

        # 数値計算: まずx方向に微分、次にy方向に微分
        dx = self.field.gradient(axis=0)
        dxdy = dx.gradient(axis=1)

        jnp.testing.assert_allclose(dxdy.data, expected_dxdy, rtol=1e-4)

        # 順序を変えても同じ結果になることを確認
        dy = self.field.gradient(axis=1)
        dydx = dy.gradient(axis=0)

        jnp.testing.assert_allclose(dxdy.data, dydx.data, rtol=1e-4)

    def test_convergence_order(self):
        """空間精度の収束次数をテスト"""
        # より細かい格子での計算
        nx2, ny2, nz2 = 32, 32, 32
        shape2 = (nx2, ny2, nz2)
        dx2 = self.Lx / (nx2 - 1)
        dy2 = self.Ly / (ny2 - 1)
        dz2 = self.Lz / (nz2 - 1)
        grid2 = GridInfo(shape=shape2, dx=(dx2, dy2, dz2))

        # 新しい格子での座標
        x2 = jnp.linspace(0, self.Lx, nx2)
        y2 = jnp.linspace(0, self.Ly, ny2)
        z2 = jnp.linspace(0, self.Lz, nz2)
        X2, Y2, Z2 = jnp.meshgrid(x2, y2, z2, indexing="ij")

        # より細かい格子でのテスト関数
        data2 = (
            jnp.sin(2 * jnp.pi * X2 / self.Lx)
            * jnp.cos(2 * jnp.pi * Y2 / self.Ly)
            * jnp.exp(-Z2 / self.Lz)
        )
        field2 = ScalarField(grid2, data2)

        # 両方の格子での勾配を計算
        grad1 = self.field.gradient(axis=0)
        grad2 = field2.gradient(axis=0)

        # 解析解
        exact1 = (
            (2 * jnp.pi / self.Lx)
            * jnp.cos(2 * jnp.pi * self.X / self.Lx)
            * jnp.cos(2 * jnp.pi * self.Y / self.Ly)
            * jnp.exp(-self.Z / self.Lz)
        )
        exact2 = (
            (2 * jnp.pi / self.Lx)
            * jnp.cos(2 * jnp.pi * X2 / self.Lx)
            * jnp.cos(2 * jnp.pi * Y2 / self.Ly)
            * jnp.exp(-Z2 / self.Lz)
        )

        # 誤差の計算
        error1 = float(jnp.max(jnp.abs(grad1.data - exact1)))
        error2 = float(jnp.max(jnp.abs(grad2.data - exact2)))

        # 収束次数の計算
        order = -jnp.log2(error2 / error1)
        self.assertGreater(order, 1.8, "空間精度が2次より低いです")
        self.assertLess(order, 2.2, "空間精度が2次より高いです")


if __name__ == "__main__":
    unittest.main()
