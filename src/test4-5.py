"""
ScalarFieldとVectorField間の相互作用をテストするモジュール

以下の演算を検証:
- ScalarField * VectorField
- VectorField * ScalarField
- ScalarField / VectorField
- VectorField / ScalarField
- 勾配・発散の連続性
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# core/fieldへのパスを追加
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.field import VectorField, ScalarField, GridInfo


class TestFieldInteractions(unittest.TestCase):
    """ScalarFieldとVectorField間の相互作用テスト"""

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

        # テストスカラー場: φ = sin(2πx/Lx)cos(2πy/Ly)exp(-z/Lz)
        self.scalar_data = (
            np.sin(2 * np.pi * self.X / self.Lx)
            * np.cos(2 * np.pi * self.Y / self.Ly)
            * np.exp(-self.Z / self.Lz)
        )
        self.scalar_field = ScalarField(self.grid, self.scalar_data)

        # テストベクトル場: u = (sin(2πx/Lx), cos(2πy/Ly), exp(-z/Lz))
        self.vector_data = [
            np.sin(2 * np.pi * self.X / self.Lx),
            np.cos(2 * np.pi * self.Y / self.Ly),
            np.exp(-self.Z / self.Lz),
        ]
        self.vector_field = VectorField(self.grid, self.vector_data)

    def test_scalar_vector_multiplication(self):
        """スカラー場とベクトル場の乗算テスト"""
        # ScalarField * VectorField
        result1 = self.scalar_field * self.vector_field
        for i in range(3):
            expected = self.scalar_data * self.vector_data[i]
            np.testing.assert_allclose(result1.components[i].data, expected)

        # VectorField * ScalarField (交換法則の確認)
        result2 = self.vector_field * self.scalar_field
        for i in range(3):
            np.testing.assert_allclose(
                result2.components[i].data, result1.components[i].data
            )

    def test_scalar_vector_division(self):
        """スカラー場によるベクトル場の除算テスト"""
        # 非ゼロのスカラー場を作成
        nonzero_scalar = ScalarField(self.grid, np.abs(self.scalar_data) + 1.0)

        # VectorField / ScalarField
        result = self.vector_field / nonzero_scalar
        for i in range(3):
            expected = self.vector_data[i] / nonzero_scalar.data
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_gradient_divergence_continuity(self):
        """スカラー場の勾配の発散がラプラシアンと一致することを確認"""
        # ∇⋅(∇φ) = ∇²φ
        grad = self.scalar_field.gradient()
        div_grad = grad.divergence()

        # 直接ラプラシアンを計算
        laplacian = self.scalar_field.divergence()

        np.testing.assert_allclose(div_grad.data, laplacian.data, rtol=1e-10)

    def test_vector_field_scalar_components(self):
        """ベクトル場の各成分をスカラー場として取得・操作"""
        for i in range(3):
            # 各成分をScalarFieldとして取得
            component = self.vector_field.components[i]
            self.assertIsInstance(component, ScalarField)

            # スカラー場との演算
            result = component * self.scalar_field
            expected = self.vector_data[i] * self.scalar_data
            np.testing.assert_allclose(result.data, expected)

    def test_gradient_curl_identity(self):
        """grad(φ)の回転が0になることを確認（数値誤差の範囲で）"""
        # ∇×(∇φ) = 0
        grad = self.scalar_field.gradient()
        curl_grad = grad.curl()

        # 結果が十分0に近いことを確認
        for component in curl_grad.components:
            self.assertTrue(np.allclose(component.data, 0.0, atol=1e-10))

    def test_vector_decomposition(self):
        """ベクトル場のヘルムホルツ分解の基本性質をテスト"""
        # 発散成分と回転成分の直交性を確認
        div = self.vector_field.divergence()
        curl = self.vector_field.curl()

        # ∇(∇⋅u)と∇×(∇×u)の内積が0になることを確認
        div_grad = div.gradient()
        curl_curl = curl.curl()

        # 内積を計算
        dot_product = div_grad.dot(curl_curl)

        # 結果が十分0に近いことを確認（境界の影響を考慮）
        interior_points = dot_product.data[1:-1, 1:-1, 1:-1]
        self.assertTrue(np.allclose(interior_points, 0.0, atol=1e-8))

    def test_mixed_operations(self):
        """複合的な演算のテスト"""
        # (φu)⋅∇φ の計算
        scalar_times_vector = self.scalar_field * self.vector_field
        grad_scalar = self.scalar_field.gradient()
        result = scalar_times_vector.dot(grad_scalar)

        # 個別の成分から計算した結果と比較
        expected = sum(
            self.scalar_data * self.vector_data[i] * grad_scalar.components[i].data
            for i in range(3)
        )
        np.testing.assert_allclose(result.data, expected)

    def test_boundary_consistency(self):
        """境界での一貫性テスト"""
        # スカラー場の勾配とベクトル場の発散が境界で矛盾しないことを確認
        grad = self.scalar_field.gradient()
        div = grad.divergence()

        # 境界でも値が発散していないことを確認
        self.assertTrue(np.all(np.isfinite(div.data)))

        # 境界での値が内部と大きく異ならないことを確認
        interior_max = np.max(np.abs(div.data[1:-1, 1:-1, 1:-1]))
        boundary_max = np.max(np.abs(div.data[0, :, :]))  # x=0面
        self.assertLess(boundary_max, 2 * interior_max)

    def test_error_handling(self):
        """エラー処理のテスト"""
        # 異なる形状のフィールド間の演算
        different_shape = (8, 8, 8)
        different_grid = GridInfo(shape=different_shape, dx=self.grid.dx)
        different_scalar = ScalarField(different_grid, np.random.rand(*different_shape))

        # 形状の不一致による例外の確認
        with self.assertRaises(ValueError):
            _ = self.vector_field * different_scalar

        with self.assertRaises(ValueError):
            _ = different_scalar * self.vector_field


if __name__ == "__main__":
    unittest.main()
