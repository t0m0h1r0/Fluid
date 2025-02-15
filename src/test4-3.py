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
import jax.numpy as jnp
from jax import random
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

        # JAXの乱数キーを生成
        key = random.PRNGKey(0)

        # テストデータの生成
        keys = random.split(key, 6)
        self.data1 = [random.uniform(keys[i], self.shape) for i in range(3)]
        self.data2 = [random.uniform(keys[i + 3], self.shape) for i in range(3)]

        # VectorFieldの作成
        self.field1 = VectorField(self.grid, self.data1)
        self.field2 = VectorField(self.grid, self.data2)

    def test_vector_addition(self):
        """ベクトル場の加算テスト"""
        result = self.field1 + self.field2
        for i in range(3):
            expected = jnp.asarray(self.data1[i] + self.data2[i])
            jnp.testing.assert_allclose(result.components[i].data, expected, rtol=1e-6)

    def test_vector_subtraction(self):
        """ベクトル場の減算テスト"""
        result = self.field1 - self.field2
        for i in range(3):
            expected = jnp.asarray(self.data1[i] - self.data2[i])
            jnp.testing.assert_allclose(result.components[i].data, expected, rtol=1e-6)

    def test_scalar_multiplication(self):
        """スカラー倍のテスト"""
        scalar = 2.5
        result = self.field1 * scalar
        for i in range(3):
            expected = jnp.asarray(self.data1[i] * scalar)
            jnp.testing.assert_allclose(result.components[i].data, expected, rtol=1e-6)

    def test_scalar_division(self):
        """スカラーによる除算テスト"""
        scalar = 2.0
        result = self.field1 / scalar
        for i in range(3):
            expected = jnp.asarray(self.data1[i] / scalar)
            jnp.testing.assert_allclose(result.components[i].data, expected, rtol=1e-6)

    def test_dot_product(self):
        """内積のテスト"""
        result = self.field1.dot(self.field2)
        expected = sum(d1 * d2 for d1, d2 in zip(self.data1, self.data2))
        jnp.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_cross_product(self):
        """外積のテスト"""
        result = self.field1.cross(self.field2)
        expected = [
            self.data1[1] * self.data2[2] - self.data1[2] * self.data2[1],
            self.data1[2] * self.data2[0] - self.data1[0] * self.data2[2],
            self.data1[0] * self.data2[1] - self.data1[1] * self.data2[0],
        ]
        for i in range(3):
            jnp.testing.assert_allclose(
                result.components[i].data, expected[i], rtol=1e-6
            )

    def test_magnitude(self):
        """大きさの計算テスト"""
        result = self.field1.magnitude()
        expected = jnp.sqrt(sum(d * d for d in self.data1))
        jnp.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_normalization(self):
        """正規化のテスト"""
        normalized = self.field1.normalize()
        magnitude = self.field1.magnitude()

        # 正規化後の大きさが1になることを確認
        normalized_magnitude = normalized.magnitude()
        jnp.testing.assert_allclose(
            normalized_magnitude.data, jnp.ones_like(magnitude.data), rtol=1e-5
        )

        # 方向が保存されていることを確認
        for i in range(3):
            expected = self.data1[i] / (magnitude.data + 1e-10)
            jnp.testing.assert_allclose(
                normalized.components[i].data, expected, rtol=1e-5
            )

    def test_vector_field_copy(self):
        """コピーのテスト"""
        copied = self.field1.copy()

        # データが同じであることを確認
        for i in range(3):
            jnp.testing.assert_allclose(
                copied.components[i].data, self.field1.components[i].data
            )

        # 独立したコピーであることを確認
        copied.components[0].data = jnp.ones_like(copied.components[0].data)
        self.assertFalse(
            jnp.array_equal(copied.components[0].data, self.field1.components[0].data)
        )

    def test_scalar_field_multiplication(self):
        """ScalarFieldとの乗算テスト"""
        scalar_data = random.uniform(random.PRNGKey(42), self.shape)
        scalar_field = ScalarField(self.grid, scalar_data)

        result = self.field1 * scalar_field
        for i in range(3):
            expected = jnp.asarray(self.data1[i] * scalar_data)
            jnp.testing.assert_allclose(result.components[i].data, expected, rtol=1e-6)

    def test_component_access(self):
        """成分アクセスのテスト"""
        # 各成分がScalarFieldとして取得できることを確認
        for i in range(3):
            component = self.field1.components[i]
            self.assertIsInstance(component, ScalarField)
            jnp.testing.assert_allclose(component.data, self.data1[i])

    def test_error_conditions(self):
        """エラー条件のテスト"""
        # 異なる形状のフィールドとの演算
        different_shape = (3, 4, 5)
        different_grid = GridInfo(shape=different_shape, dx=self.dx)
        different_field = VectorField(
            different_grid,
            [random.uniform(random.PRNGKey(i), different_shape) for i in range(3)],
        )

        with self.assertRaises(ValueError):
            _ = self.field1 + different_field

        with self.assertRaises(ValueError):
            _ = self.field1.dot(different_field)

    def test_field_info_preservation(self):
        """フィールド情報の保存をテスト"""
        # 演算後にグリッド情報が維持されることを確認
        result = self.field1 + self.field2
        self.assertEqual(result.grid, self.grid)
        self.assertEqual(result.shape, self.grid.shape)
        self.assertEqual(result.dx, self.grid.dx)


if __name__ == "__main__":
    unittest.main()
