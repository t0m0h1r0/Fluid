"""
VectorFieldの基本演算をテストするモジュール

以下の演算を検証:
- 四則演算 (+, -, *, /)
- スカラー倍
- 内積・外積
- ノルム・正規化
- magnitude計算
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# core/fieldへのパスを追加
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.field import VectorField, ScalarField, GridInfo


class TestVectorFieldBasic(unittest.TestCase):
    """VectorFieldの基本演算テスト"""

    def setUp(self):
        """テストの前準備"""
        # 3D格子の設定
        self.shape = (4, 5, 6)
        self.dx = (0.1, 0.1, 0.1)
        self.grid = GridInfo(shape=self.shape, dx=self.dx)

        # テストデータの生成（3成分のベクトル場）
        self.data1 = [
            np.random.rand(*self.shape),
            np.random.rand(*self.shape),
            np.random.rand(*self.shape),
        ]
        self.data2 = [
            np.random.rand(*self.shape),
            np.random.rand(*self.shape),
            np.random.rand(*self.shape),
        ]

        # VectorFieldの作成
        self.field1 = VectorField(self.grid, self.data1)
        self.field2 = VectorField(self.grid, self.data2)

    def test_vector_addition(self):
        """ベクトル場の加算テスト"""
        result = self.field1 + self.field2
        for i in range(3):
            expected = self.data1[i] + self.data2[i]
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_vector_subtraction(self):
        """ベクトル場の減算テスト"""
        result = self.field1 - self.field2
        for i in range(3):
            expected = self.data1[i] - self.data2[i]
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_scalar_multiplication(self):
        """スカラー倍のテスト"""
        scalar = 2.5
        result = self.field1 * scalar
        for i in range(3):
            expected = self.data1[i] * scalar
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_scalar_division(self):
        """スカラーによる除算テスト"""
        scalar = 2.0
        result = self.field1 / scalar
        for i in range(3):
            expected = self.data1[i] / scalar
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_dot_product(self):
        """内積のテスト"""
        result = self.field1.dot(self.field2)
        expected = sum(self.data1[i] * self.data2[i] for i in range(3))
        np.testing.assert_allclose(result.data, expected)

    def test_cross_product(self):
        """外積のテスト"""
        result = self.field1.cross(self.field2)
        expected = [
            self.data1[1] * self.data2[2] - self.data1[2] * self.data2[1],
            self.data1[2] * self.data2[0] - self.data1[0] * self.data2[2],
            self.data1[0] * self.data2[1] - self.data1[1] * self.data2[0],
        ]
        for i in range(3):
            np.testing.assert_allclose(result.components[i].data, expected[i])

    def test_magnitude(self):
        """大きさの計算テスト"""
        result = self.field1.magnitude()
        expected = np.sqrt(sum(d * d for d in self.data1))
        np.testing.assert_allclose(result.data, expected)

    def test_normalization(self):
        """正規化のテスト"""
        normalized = self.field1.normalize()
        magnitude = self.field1.magnitude()

        # 正規化後の大きさが1になることを確認
        normalized_magnitude = normalized.magnitude()
        np.testing.assert_allclose(
            normalized_magnitude.data, np.ones_like(magnitude.data), rtol=1e-5
        )

        # 方向が保存されていることを確認
        for i in range(3):
            expected = self.data1[i] / (magnitude.data + 1e-6)
            np.testing.assert_allclose(
                normalized.components[i].data, expected, rtol=1e-5
            )

    def test_vector_field_copy(self):
        """コピーのテスト"""
        copied = self.field1.copy()

        # 同じ値を持つことを確認
        for i in range(3):
            np.testing.assert_allclose(
                copied.components[i].data, self.field1.components[i].data
            )

        # 独立したコピーであることを確認
        copied.components[0].data[0, 0, 0] = 999.9
        self.assertNotAlmostEqual(
            copied.components[0].data[0, 0, 0], self.field1.components[0].data[0, 0, 0]
        )

    def test_scalar_field_multiplication(self):
        """ScalarFieldとの乗算テスト"""
        scalar_data = np.random.rand(*self.shape)
        scalar_field = ScalarField(self.grid, scalar_data)

        result = self.field1 * scalar_field
        for i in range(3):
            expected = self.data1[i] * scalar_data
            np.testing.assert_allclose(result.components[i].data, expected)

    def test_component_access(self):
        """成分アクセスのテスト"""
        # 各成分がScalarFieldとして取得できることを確認
        for i in range(3):
            component = self.field1.components[i]
            self.assertIsInstance(component, ScalarField)
            np.testing.assert_allclose(component.data, self.data1[i])

    def test_error_conditions(self):
        """エラー条件のテスト"""
        # 異なる形状のフィールドとの演算
        different_shape = (3, 4, 5)
        different_grid = GridInfo(shape=different_shape, dx=self.dx)
        different_field = VectorField(
            different_grid, [np.random.rand(*different_shape) for _ in range(3)]
        )

        with self.assertRaises(ValueError):
            _ = self.field1 + different_field

        with self.assertRaises(ValueError):
            _ = self.field1.dot(different_field)


if __name__ == "__main__":
    unittest.main()
