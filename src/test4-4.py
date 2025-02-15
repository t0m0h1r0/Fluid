"""
VectorFieldの微分演算をテストするモジュール

以下の演算を検証:
- 勾配 (gradient)
- 発散 (divergence)
- 回転 (curl)
- 対称勾配テンソル
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# core/fieldへのパスを追加
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.field import VectorField, GridInfo


class TestVectorFieldDerivatives(unittest.TestCase):
    """VectorFieldの微分演算テスト"""

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
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        z = np.linspace(0, self.Lz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        # テストベクトル場の設定（解析的に微分可能な関数）
        # u = (sin(2πx/Lx)cos(2πy/Ly), cos(2πx/Lx)sin(2πy/Ly), exp(-z/Lz))
        self.data = [
            np.sin(2 * np.pi * self.X / self.Lx) * np.cos(2 * np.pi * self.Y / self.Ly),
            np.cos(2 * np.pi * self.X / self.Lx) * np.sin(2 * np.pi * self.Y / self.Ly),
            np.exp(-self.Z / self.Lz),
        ]
        self.field = VectorField(self.grid, self.data)

    def test_divergence(self):
        """発散のテスト: ∇⋅u"""
        # 発散の解析解: ∂u/∂x + ∂v/∂y + ∂w/∂z
        expected_div = (
            # ∂u/∂x
            (2 * np.pi / self.Lx)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly)
            +
            # ∂v/∂y
            (2 * np.pi / self.Ly)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly)
            +
            # ∂w/∂z
            (-1 / self.Lz) * np.exp(-self.Z / self.Lz)
        )

        div = self.field.divergence()
        np.testing.assert_allclose(div.data, expected_div, rtol=1e-2)

    def test_curl(self):
        """回転のテスト: ∇×u"""
        # 回転の解析解
        expected_curl = [
            # (∂w/∂y - ∂v/∂z)
            (-2 * np.pi / self.Ly)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly),
            # (∂u/∂z - ∂w/∂x)
            (-2 * np.pi / self.Lx)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly),
            # (∂v/∂x - ∂u/∂y)
            (-4 * np.pi**2 / (self.Lx * self.Ly))
            * np.sin(2 * np.pi * self.X / self.Lx)
            * np.sin(2 * np.pi * self.Y / self.Ly),
        ]

        curl = self.field.curl()
        for i in range(3):
            np.testing.assert_allclose(
                curl.components[i].data, expected_curl[i], rtol=1e-2
            )

    def test_symmetric_gradient(self):
        """対称勾配テンソルのテスト: ∇ᵤₛ = 0.5(∇u + ∇uᵀ)"""
        sym_grad = self.field.symmetric_gradient()

        # 対称性のチェック
        for i in range(3):
            for j in range(3):
                # i方向のj成分の微分と、j方向のi成分の微分が等しいことを確認
                grad_i = self.field.components[i].gradient(j)
                grad_j = self.field.components[j].gradient(i)
                expected = 0.5 * (grad_i + grad_j)

                # テンソルの(i,j)成分の取得
                tensor_comp = sym_grad.components[i * 3 + j]
                np.testing.assert_allclose(tensor_comp.data, expected, rtol=1e-2)

    def test_gradient(self):
        """勾配のテスト（各成分について）"""
        grad = self.field.gradient()

        # x成分の勾配
        expected_grad_x = [
            # ∂u/∂x
            (2 * np.pi / self.Lx)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly),
            # ∂u/∂y
            (-2 * np.pi / self.Ly)
            * np.sin(2 * np.pi * self.X / self.Lx)
            * np.sin(2 * np.pi * self.Y / self.Ly),
            # ∂u/∂z
            np.zeros_like(self.X),
        ]

        # y成分の勾配
        expected_grad_y = [
            # ∂v/∂x
            (-2 * np.pi / self.Lx)
            * np.sin(2 * np.pi * self.X / self.Lx)
            * np.sin(2 * np.pi * self.Y / self.Ly),
            # ∂v/∂y
            (2 * np.pi / self.Ly)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly),
            # ∂v/∂z
            np.zeros_like(self.X),
        ]

        # z成分の勾配
        expected_grad_z = [
            # ∂w/∂x
            np.zeros_like(self.X),
            # ∂w/∂y
            np.zeros_like(self.X),
            # ∂w/∂z
            (-1 / self.Lz) * np.exp(-self.Z / self.Lz),
        ]

        expected_grads = [expected_grad_x, expected_grad_y, expected_grad_z]

        # 各成分のチェック
        for i in range(3):
            grad_component = grad.components[i]
            for j in range(3):
                np.testing.assert_allclose(
                    grad_component.gradient(j).data, expected_grads[i][j], rtol=1e-2
                )

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
        x2 = np.linspace(0, self.Lx, nx2)
        y2 = np.linspace(0, self.Ly, ny2)
        z2 = np.linspace(0, self.Lz, nz2)
        X2, Y2, Z2 = np.meshgrid(x2, y2, z2, indexing="ij")

        # より細かい格子でのテストベクトル場
        data2 = [
            np.sin(2 * np.pi * X2 / self.Lx) * np.cos(2 * np.pi * Y2 / self.Ly),
            np.cos(2 * np.pi * X2 / self.Lx) * np.sin(2 * np.pi * Y2 / self.Ly),
            np.exp(-Z2 / self.Lz),
        ]
        field2 = VectorField(grid2, data2)

        # 両方の格子での発散を計算
        div1 = self.field.divergence()
        div2 = field2.divergence()

        # 解析解
        expected_div1 = (
            (2 * np.pi / self.Lx)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly)
            + (2 * np.pi / self.Ly)
            * np.cos(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly)
            + (-1 / self.Lz) * np.exp(-self.Z / self.Lz)
        )
        expected_div2 = (
            (2 * np.pi / self.Lx)
            * np.cos(2 * np.pi * X2 / self.Lx)
            * np.cos(2 * np.pi * Y2 / self.Ly)
            + (2 * np.pi / self.Ly)
            * np.cos(2 * np.pi * X2 / self.Lx)
            * np.cos(2 * np.pi * Y2 / self.Ly)
            + (-1 / self.Lz) * np.exp(-Z2 / self.Lz)
        )

        # 誤差の計算
        error1 = np.max(np.abs(div1.data - expected_div1))
        error2 = np.max(np.abs(div2.data - expected_div2))

        # 収束次数の計算
        order = -np.log2(error2 / error1)
        self.assertGreater(order, 1.8, "空間精度が2次より低いです")
        self.assertLess(order, 2.2, "空間精度が2次より高いです")


if __name__ == "__main__":
    unittest.main()
