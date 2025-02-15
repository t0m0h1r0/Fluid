import numpy as np
from core.field import GridInfo, ScalarField, VectorField, FieldFactory
from typing import Tuple, Optional, List, Union, Callable


class FieldValidator:
    """
    ScalarFieldとVectorFieldの計算結果を検証するクラス。
    NumPyでの直接計算結果と比較して、計算の一貫性を確認します。
    """

    def __init__(
        self, 
        shape: Tuple[int, int, int] = (32, 32, 32), 
        dx: Optional[Tuple[float, float, float]] = None,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ):
        """
        検証ツールを初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔。指定されない場合は均一間隔を使用
            rtol: 相対許容誤差
            atol: 絶対許容誤差
        """
        self.shape = shape
        self.dx = dx or tuple(1.0 / (s - 1) for s in shape)
        self.grid_info = GridInfo(shape=shape, dx=self.dx)
        self.rtol = rtol
        self.atol = atol

    def _generate_test_data(
        self, 
        data_type: str = 'random', 
        function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None
    ) -> Union[np.ndarray, ScalarField, VectorField]:
        """
        テスト用のデータを生成

        Args:
            data_type: データの種類 ('random', 'sine', 'custom')
            function: カスタム関数（data_type='custom'の場合に使用）

        Returns:
            生成されたデータ
        """
        # グリッドの座標を生成
        coords = np.meshgrid(
            np.linspace(0, 1, self.shape[0]),
            np.linspace(0, 1, self.shape[1]),
            np.linspace(0, 1, self.shape[2]),
            indexing='ij'
        )

        if data_type == 'random':
            return np.random.rand(*self.shape)
        elif data_type == 'sine':
            # 3次元正弦波
            return np.sin(2 * np.pi * coords[0]) * \
                   np.cos(2 * np.pi * coords[1]) * \
                   np.sin(2 * np.pi * coords[2])
        elif data_type == 'custom' and function is not None:
            return function(*coords)
        else:
            raise ValueError("Invalid data type or no custom function provided")

    def validate_gradient(self, verbose: bool = True) -> bool:
        """
        勾配計算の検証

        Args:
            verbose: 詳細な情報を表示するかどうか

        Returns:
            検証結果（True: 成功、False: 失敗）
        """
        # テストデータの生成
        test_data = self._generate_test_data('sine')
        scalar_field = ScalarField(self.grid_info, test_data)

        # NumPyでの勾配計算
        numpy_gradients = [
            np.gradient(test_data, self.dx[i], axis=i) 
            for i in range(3)
        ]

        # ScalarFieldでの勾配計算
        field_gradients = [
            scalar_field.gradient(i).data 
            for i in range(3)
        ]

        # 検証
        all_valid = True
        for i, (np_grad, field_grad) in enumerate(zip(numpy_gradients, field_gradients)):
            is_valid = np.allclose(
                np_grad, field_grad, 
                rtol=self.rtol, atol=self.atol
            )
            all_valid &= is_valid

            if verbose and not is_valid:
                print(f"Gradient in direction {i} failed validation")
                print(f"Max absolute difference: {np.max(np.abs(np_grad - field_grad))}")

        return all_valid

    def validate_divergence(self, verbose: bool = True) -> bool:
        """
        発散計算の検証

        Args:
            verbose: 詳細な情報を表示するかどうか

        Returns:
            検証結果（True: 成功、False: 失敗）
        """
        # ベクトル場の生成
        test_data = [
            self._generate_test_data('sine') 
            for _ in range(3)
        ]
        vector_field = VectorField(self.grid_info, tuple(test_data))

        # NumPyでの発散計算
        numpy_div = np.zeros(self.shape)
        for i in range(3):
            numpy_div += np.gradient(test_data[i], self.dx[i], axis=i)

        # ScalarFieldでの発散計算
        field_div = vector_field.divergence().data

        # 検証
        is_valid = np.allclose(
            numpy_div, field_div, 
            rtol=self.rtol, atol=self.atol
        )

        if verbose and not is_valid:
            print("Divergence failed validation")
            print(f"Max absolute difference: {np.max(np.abs(numpy_div - field_div))}")

        return is_valid

    def validate_curl(self, verbose: bool = True) -> bool:
        """
        回転（カール）計算の検証

        Args:
            verbose: 詳細な情報を表示するかどうか

        Returns:
            検証結果（True: 成功、False: 失敗）
        """
        # ベクトル場の生成
        test_data = [
            self._generate_test_data('sine') 
            for _ in range(3)
        ]
        vector_field = VectorField(self.grid_info, tuple(test_data))

        # NumPyでの回転計算
        numpy_curl = [
            np.gradient(test_data[2], self.dx[1], axis=1) - 
            np.gradient(test_data[1], self.dx[2], axis=2),
            np.gradient(test_data[0], self.dx[2], axis=2) - 
            np.gradient(test_data[2], self.dx[0], axis=0),
            np.gradient(test_data[1], self.dx[0], axis=0) - 
            np.gradient(test_data[0], self.dx[1], axis=1)
        ]

        # VectorFieldでの回転計算
        field_curl = [comp.data for comp in vector_field.curl().components]

        # 検証
        all_valid = True
        for i, (np_curl, field_curl_comp) in enumerate(zip(numpy_curl, field_curl)):
            is_valid = np.allclose(
                np_curl, field_curl_comp, 
                rtol=self.rtol, atol=self.atol
            )
            all_valid &= is_valid

            if verbose and not is_valid:
                print(f"Curl in direction {i} failed validation")
                print(f"Max absolute difference: {np.max(np.abs(np_curl - field_curl_comp))}")

        return all_valid

    def validate_symmetric_gradient(self, verbose: bool = True) -> bool:
        """
        対称勾配テンソルの計算を検証

        Args:
            verbose: 詳細な情報を表示するかどうか

        Returns:
            検証結果（True: 成功、False: 失敗）
        """
        # ベクトル場の生成
        test_data = [
            self._generate_test_data('sine') 
            for _ in range(3)
        ]
        vector_field = VectorField(self.grid_info, tuple(test_data))

        # NumPyでの対称勾配テンソルの計算
        numpy_sym_grad = []
        for i in range(3):
            sym_grad_row = []
            for j in range(3):
                # 対称勾配: 0.5(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
                dui_dxj = np.gradient(test_data[i], self.dx[j], axis=j)
                duj_dxi = np.gradient(test_data[j], self.dx[i], axis=i)
                sym_grad_ij = 0.5 * (dui_dxj + duj_dxi)
                sym_grad_row.append(sym_grad_ij)
            numpy_sym_grad.extend(sym_grad_row[:3])

        # VectorFieldでの対称勾配テンソルの計算
        field_sym_grad = [
            comp.data for comp in vector_field.symmetric_gradient().components
        ]

        # 検証
        all_valid = True
        for i, (np_sym_grad, field_sym_grad_comp) in enumerate(zip(numpy_sym_grad, field_sym_grad)):
            
            # より詳細な統計情報を収集
            abs_diff = np.abs(np_sym_grad - field_sym_grad_comp)
            relative_diff = abs_diff / (np.abs(np_sym_grad) + self.atol)
            
            is_valid = np.allclose(
                np_sym_grad, field_sym_grad_comp, 
                rtol=self.rtol, atol=self.atol
            )
            all_valid &= is_valid

            if verbose and not is_valid:
                print(f"\nSymmetric Gradient in direction {i} failed validation:")
                print(f"Max absolute difference: {np.max(abs_diff)}")
                print(f"Max relative difference: {np.max(relative_diff)}")
                
                # デバッグ情報の追加
                print("\nNumPy Calculation:")
                print(f"Min: {np.min(np_sym_grad)}")
                print(f"Max: {np.max(np_sym_grad)}")
                print(f"Mean: {np.mean(np_sym_grad)}")
                print(f"Std: {np.std(np_sym_grad)}")
                
                print("\nField Calculation:")
                print(f"Min: {np.min(field_sym_grad_comp)}")
                print(f"Max: {np.max(field_sym_grad_comp)}")
                print(f"Mean: {np.mean(field_sym_grad_comp)}")
                print(f"Std: {np.std(field_sym_grad_comp)}")

        return all_valid

    def run_validation_suite(self) -> bool:
        """
        全ての検証テストを実行

        Returns:
            全てのテストが成功したかどうか
        """
        print("Running Field Validation Suite...")
        
        tests = [
            ("Gradient", self.validate_gradient),
            ("Divergence", self.validate_divergence),
            ("Curl", self.validate_curl),
            ("Symmetric Gradient", self.validate_symmetric_gradient)
        ]

        all_tests_passed = True
        for test_name, test_method in tests:
            print(f"\nValidating {test_name}...")
            test_result = test_method()
            if test_result:
                print(f"{test_name} ✅ PASSED")
            else:
                print(f"{test_name} ❌ FAILED")
                all_tests_passed = False

        print("\nValidation Suite Complete.")
        return all_tests_passed


def main():
    """
    検証ツールのメインエントリーポイント
    """
    validator = FieldValidator(
        shape=(32, 32, 32),  # デフォルトのグリッドサイズ
        rtol=1e-5,  # 相対許容誤差
        atol=1e-8   # 絶対許容誤差
    )

    result = validator.run_validation_suite()
    
    if result:
        print("\n🎉 全ての検証テストに合格しました！")
        exit(0)
    else:
        print("\n❌ 一部または全ての検証テストで問題が見つかりました。")
        exit(1)


if __name__ == "__main__":
    main()