"""
ScalarFieldの基本演算をテストするモジュール

以下の演算を検証:
- 四則演算 (+, -, *, /)
- 累乗 (**)
- 単項演算子 (-)
- 統計量 (min, max, mean, sum)
- 積分
"""

import unittest
import numpy as np
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

        # テストデータの生成
        self.data1 = np.random.rand(*self.shape)
        self.data2 = np.random.rand(*self.shape)

        # ScalarFieldの作成
        self.field1 = ScalarField(self.grid, self.data1)
        self.field2 = ScalarField(self.grid, self.data2)

    def test_addition(self):
        """加算のテスト"""
        # ScalarField + ScalarField
        result = self.field1 + self.field2
        expected = self.data1 + self.data2
        np.testing.assert_allclose(result.data, expected)

        # ScalarField + float
        scalar = 2.5
        result = self.field1 + scalar
        expected = self.data1 + scalar
        np.testing.assert_allclose(result.data, expected)

    def test_subtraction(self):
        """減算のテスト"""
        # ScalarField - ScalarField
        result = self.field1 - self.field2
        expected = self.data1 - self.data2
        np.testing.assert_allclose(result.data, expected)

        # ScalarField - float
        scalar = 1.5
        result = self.field1 - scalar
        expected = self.data1 - scalar
        np.testing.assert_allclose(result.data, expected)

    def test_multiplication(self):
        """乗算のテスト"""
        # ScalarField * ScalarField
        result = self.field1 * self.field2
        expected = self.data1 * self.data2
        np.testing.assert_allclose(result.data, expected)

        # ScalarField * float
        scalar = 3.0
        result = self.field1 * scalar
        expected = self.data1 * scalar
        np.testing.assert_allclose(result.data, expected)

    def test_division(self):
        """除算のテスト"""
        # ScalarField / ScalarField
        result = self.field1 / (self.field2 + 1e-10)  # ゼロ除算を防ぐ
        expected = self.data1 / (self.data2 + 1e-10)
        np.testing.assert_allclose(result.data, expected)

        # ScalarField / float
        scalar = 2.0
        result = self.field1 / scalar
        expected = self.data1 / scalar
        np.testing.assert_allclose(result.data, expected)

    def test_power(self):
        """累乗のテスト"""
        # ScalarField ** 2
        power = 2
        result = self.field1**power
        expected = self.data1**power
        np.testing.assert_allclose(result.data, expected)

        # ScalarField ** 0.5
        power = 0.5
        result = self.field1**power
        expected = self.data1**power
        np.testing.assert_allclose(result.data, expected)

    def test_negation(self):
        """単項マイナスのテスト"""
        result = -self.field1
        expected = -self.data1
        np.testing.assert_allclose(result.data, expected)

    def test_statistical_operations(self):
        """統計演算のテスト"""
        # 最小値
        self.assertAlmostEqual(self.field1.min(), np.min(self.data1))

        # 最大値
        self.assertAlmostEqual(self.field1.max(), np.max(self.data1))

        # 平均値
        self.assertAlmostEqual(self.field1.mean(), np.mean(self.data1))

        # 合計値
        self.assertAlmostEqual(self.field1.sum(), np.sum(self.data1))

    def test_integration(self):
        """積分のテスト"""
        # 体積積分のテスト
        dv = np.prod(self.dx)  # 体積要素
        expected_integral = np.sum(self.data1) * dv
        computed_integral = self.field1.integrate()
        self.assertAlmostEqual(computed_integral, expected_integral)

    def test_field_equality(self):
        """等価性のテスト"""
        # 同じデータを持つフィールド
        field3 = ScalarField(self.grid, self.data1.copy())
        self.assertEqual(self.field1, field3)

        # 異なるデータを持つフィールド
        self.assertNotEqual(self.field1, self.field2)


if __name__ == "__main__":
    unittest.main()
