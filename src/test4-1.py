import unittest
import numpy as np
from jax import random
from pathlib import Path
import sys

# core/fieldへのパスを追加
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from core.field import ScalarField, GridInfo


class TestScalarFieldBasic(unittest.TestCase):
    """ScalarFieldの基本演算テスト"""

    def setUp(self):
        """テストの前準備"""
        # 3D格子の設定
        self.shape = (4, 5, 6)
        self.dx = (0.1, 0.1, 0.1)
        self.grid = GridInfo(shape=self.shape, dx=self.dx)

        # JAXの乱数キーを生成
        key = random.PRNGKey(0)

        # テストデータの生成
        self.data1 = random.uniform(key, shape=self.shape)
        key, subkey = random.split(key)
        self.data2 = random.uniform(subkey, shape=self.shape)

        # ScalarFieldの作成
        self.field1 = ScalarField(self.grid, self.data1)
        self.field2 = ScalarField(self.grid, self.data2)

    def test_addition(self):
        """加算のテスト"""
        # ScalarField + ScalarField
        result = self.field1 + self.field2
        expected = self.data1 + self.data2
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

        # ScalarField + float
        scalar = 2.5
        result = self.field1 + scalar
        expected = self.data1 + scalar
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_subtraction(self):
        """減算のテスト"""
        # ScalarField - ScalarField
        result = self.field1 - self.field2
        expected = self.data1 - self.data2
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

        # ScalarField - float
        scalar = 1.5
        result = self.field1 - scalar
        expected = self.data1 - scalar
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_multiplication(self):
        """乗算のテスト"""
        # ScalarField * ScalarField
        result = self.field1 * self.field2
        expected = self.data1 * self.data2
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

        # ScalarField * float
        scalar = 3.0
        result = self.field1 * scalar
        expected = self.data1 * scalar
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_division(self):
        """除算のテスト"""
        # ScalarField / ScalarField
        # ゼロ除算を防ぐため、分母に小さな値を加える
        epsilon = 1e-10
        result = self.field1 / (self.field2 + epsilon)
        expected = self.data1 / (self.data2 + epsilon)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

        # ScalarField / float
        scalar = 2.0
        result = self.field1 / scalar
        expected = self.data1 / scalar
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_power(self):
        """累乗のテスト"""
        # ScalarField ** 2
        power = 2
        result = self.field1**power
        expected = np.power(np.abs(self.data1), power)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

        # ScalarField ** 0.5
        power = 0.5
        # 負の値を避けるため、絶対値を取る
        result = self.field1.abs() ** power
        expected = np.power(np.abs(self.data1), power)
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_negation(self):
        """単項マイナスのテスト"""
        result = -self.field1
        expected = -self.data1
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)

    def test_statistical_operations(self):
        """統計演算のテスト"""
        # 最小値
        self.assertAlmostEqual(
            float(self.field1.min()), float(np.min(self.data1)), places=6
        )

        # 最大値
        self.assertAlmostEqual(
            float(self.field1.max()), float(np.max(self.data1)), places=6
        )

        # 平均値
        self.assertAlmostEqual(
            float(self.field1.mean()), float(np.mean(self.data1)), places=6
        )

        # 合計値
        self.assertAlmostEqual(
            float(self.field1.sum()), float(np.sum(self.data1)), places=6
        )

    def test_integration(self):
        """積分のテスト"""
        # 体積積分のテスト
        dv = np.prod(np.array(self.dx))  # 体積要素
        expected_integral = float(np.sum(self.data1)) * dv
        computed_integral = self.field1.integrate()
        self.assertAlmostEqual(computed_integral, expected_integral, places=6)

    def test_field_equality(self):
        """等価性のテスト"""
        # 同じデータを持つフィールド
        field3 = ScalarField(self.grid, self.data1)
        self.assertEqual(self.field1, field3)

        # 異なるデータを持つフィールド
        self.assertNotEqual(self.field1, self.field2)

    def test_grid_preservation(self):
        """グリッド情報の保存をテスト"""
        # 演算後にグリッド情報が維持されることを確認
        result = self.field1 + self.field2
        self.assertEqual(result.grid, self.grid)
        self.assertEqual(result.shape, self.grid.shape)
        self.assertEqual(result.dx, self.grid.dx)

    def test_scalar_field_copy(self):
        """コピーのテスト"""
        copied = self.field1.copy()

        # データが同じであることを確認
        np.testing.assert_allclose(copied.data, self.field1.data)

        # 独立したコピーであることを確認
        # 新しい配列で上書きする
        copied = ScalarField(self.grid, np.zeros_like(copied.data))
        self.assertFalse(np.array_equal(copied.data, self.field1.data))


if __name__ == "__main__":
    unittest.main()
