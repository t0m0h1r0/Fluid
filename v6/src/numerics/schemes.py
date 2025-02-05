# numerics/schemes.py
import numpy as np
from core.interfaces import NumericalScheme, BoundaryCondition
from typing import List, Tuple, Optional

class CompactFiniteDifferenceScheme(NumericalScheme):
    """
    4次精度コンパクト差分スキーム
    
    高精度の空間微分と時間発展のための数値スキーム
    """
    def __init__(self, 
                 order: int = 4, 
                 time_integration: str = 'explicit'):
        """
        コンパクト差分スキームの初期化
        
        Args:
            order: 精度の次数（現在は4固定）
            time_integration: 時間発展の方法 ('explicit', 'implicit')
        """
        self.order = order
        self.time_integration = time_integration
        
        # 数値安定化パラメータ
        self.epsilon = 1e-10  # 数値安定化のための微小定数
    
    def discretize(self, 
                   flux: np.ndarray, 
                   source: np.ndarray,
                   state: np.ndarray, 
                   dt: float) -> np.ndarray:
        """
        フラックスとソース項を用いて状態を離散化
        
        Args:
            flux: 流束
            source: ソース項
            state: 現在の状態
            dt: 時間刻み
        
        Returns:
            更新された状態
        """
        if self.time_integration == 'explicit':
            # 陽的オイラー法
            return self._explicit_time_integration(flux, source, state, dt)
        elif self.time_integration == 'implicit':
            # 陰的オイラー法（単純な実装）
            return self._implicit_time_integration(flux, source, state, dt)
        else:
            raise ValueError(f"未サポートの時間発展法: {self.time_integration}")
    
    def _explicit_time_integration(self, 
                                   flux: np.ndarray, 
                                   source: np.ndarray,
                                   state: np.ndarray, 
                                   dt: float) -> np.ndarray:
        """
        陽的オイラー法による時間発展
        
        Args:
            flux: 流束
            source: ソース項
            state: 現在の状態
            dt: 時間刻み
        
        Returns:
            更新された状態
        """
        return state - dt * (flux + source)
    
    def _implicit_time_integration(self, 
                                   flux: np.ndarray, 
                                   source: np.ndarray,
                                   state: np.ndarray, 
                                   dt: float) -> np.ndarray:
        """
        陰的オイラー法による時間発展
        
        Args:
            flux: 流束
            source: ソース項
            state: 現在の状態
            dt: 時間刻み
        
        Returns:
            更新された状態
        """
        # 簡易的な陰解法（線形化）
        return state - dt * (flux + source) / (1 + dt)
    
    def compute_gradient(self, 
                         field: np.ndarray, 
                         axis: int, 
                         boundary_condition: BoundaryCondition) -> np.ndarray:
        """
        4次精度コンパクト差分による勾配計算
        
        Args:
            field: 入力フィールド
            axis: 勾配を計算する軸
            boundary_condition: 境界条件
        
        Returns:
            勾配場
        """
        # 多次元配列に対する勾配計算
        gradient = np.zeros_like(field)
        
        # 各軸に垂直な方向でループ
        for indices in self._get_orthogonal_indices(field.shape, axis):
            # 指定された軸に沿った1次元配列を取得
            line = self._get_line(field, axis, indices)
            
            # 境界条件を考慮した勾配計算
            grad_line = self._compute_compact_gradient(line)
            
            # 結果を勾配配列に設定
            self._set_line(gradient, axis, indices, grad_line)
        
        return gradient
    
    def compute_laplacian(self, 
                          field: np.ndarray, 
                          boundary_condition: BoundaryCondition) -> np.ndarray:
        """
        4次精度コンパクト差分によるラプラシアン計算
        
        Args:
            field: 入力フィールド
            boundary_condition: 境界条件
        
        Returns:
            ラプラシアン
        """
        laplacian = np.zeros_like(field)
        
        # 各軸に対してラプラシアンを計算
        for axis in range(field.ndim):
            # 各軸に垂直な方向でループ
            for indices in self._get_orthogonal_indices(field.shape, axis):
                # 指定された軸に沿った1次元配列を取得
                line = self._get_line(field, axis, indices)
                
                # 4次精度中心差分によるラプラシアン計算
                laplacian_line = self._compute_compact_laplacian(line)
                
                # 結果をラプラシアン配列に設定
                self._set_line(laplacian, axis, indices, laplacian_line)
        
        return laplacian
    
    def _compute_compact_gradient(self, line: np.ndarray) -> np.ndarray:
        """
        1次元配列に対する4次精度コンパクト差分勾配
        
        Args:
            line: 1次元入力配列
        
        Returns:
            勾配
        """
        n = len(line)
        
        # 配列が小さすぎる場合は通常の中心差分
        if n < 5:
            return np.gradient(line)
        
        # 4次精度中心差分
        grad_line = np.zeros_like(line)
        grad_line[2:-2] = (
            -line[4:] + 8*line[3:-1] - 8*line[1:-3] + line[:-4]
        ) / 12.0
        
        # 境界点の処理（4次精度片側差分）
        grad_line[0] = (-25*line[0] + 48*line[1] - 36*line[2] + 16*line[3] - 3*line[4]) / 12
        grad_line[1] = (-3*line[0] - 10*line[1] + 18*line[2] - 6*line[3] + line[4]) / 12
        grad_line[-1] = (25*line[-1] - 48*line[-2] + 36*line[-3] - 16*line[-4] + 3*line[-5]) / 12
        grad_line[-2] = (3*line[-1] + 10*line[-2] - 18*line[-3] + 6*line[-4] - line[-5]) / 12
        
        return grad_line
    
    def _compute_compact_laplacian(self, line: np.ndarray) -> np.ndarray:
        """
        1次元配列に対する4次精度コンパクト差分ラプラシアン
        
        Args:
            line: 1次元入力配列
        
        Returns:
            ラプラシアン
        """
        n = len(line)
        
        # 配列が小さすぎる場合は単純な2次差分
        if n < 5:
            return np.gradient(np.gradient(line))
        
        # 4次精度中心差分によるラプラシアン
        laplacian_line = np.zeros_like(line)
        laplacian_line[2:-2] = (
            -line[4:] + 16*line[3:-1] - 30*line[2:-2] + 
            16*line[1:-3] - line[:-4]
        ) / 12.0
        
        # 境界点の特別な処理
        # 前方境界
        laplacian_line[0] = (
            -49*line[0] + 96*line[1] - 
            54*line[2] + 16*line[3] - 
            3*line[4]
        ) / 12.0
        
        laplacian_line[1] = (
            -3*line[0] - 10*line[1] + 
            18*line[2] - 6*line[3] + 
            line[4]
        ) / 12.0
        
        # 後方境界
        laplacian_line[-1] = (
            49*line[-1] - 96*line[-2] + 
            54*line[-3] - 16*line[-4] + 
            3*line[-5]
        ) / 12.0
        
        laplacian_line[-2] = (
            3*line[-1] + 10*line[-2] - 
            18*line[-3] + 6*line[-4] - 
            line[-5]
        ) / 12.0
        
        return laplacian_line
    
    def _get_orthogonal_indices(self, 
                                shape: Tuple[int, ...], 
                                axis: int) -> List[Tuple]:
        """
        指定された軸に垂直な方向のインデックスを生成
        
        Args:
            shape: 配列の形状
            axis: 勾配を計算する軸
        
        Returns:
            直交する方向のインデックスのリスト
        """
        # 軸以外の次元のインデックスを生成
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        
        # すべての組み合わせを生成
        from itertools import product
        return list(product(*ranges))
    
    def _get_line(self, array: np.ndarray, axis: int, indices: Tuple) -> np.ndarray:
        """
        指定された軸に沿ったラインを取得
        
        Args:
            array: 入力配列
            axis: 抽出する軸
            indices: 他の軸のインデックス
        
        Returns:
            抽出された1次元配列
        """
        # インデックスのリストを作成
        idx_list = list(indices)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, 
                  array: np.ndarray, 
                  axis: int, 
                  indices: Tuple, 
                  values: np.ndarray):
        """
        指定された軸に沿ってラインを設定
        
        Args:
            array: 更新する配列
            axis: 設定する軸
            indices: 他の軸のインデックス
            values: 設定する値
        """
        idx_list = list(indices)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values

# デモンストレーション関数
def demonstrate_compact_scheme():
    """
    コンパクト差分スキームのデモンストレーション
    """
    # 3D配列の作成
    shape = (32, 32, 64)
    field = np.random.rand(*shape)
    
    # スキームのインスタンス作成（デフォルトの境界条件を仮定）
    class DummyBoundaryCondition:
        def apply(self, state):
            return state
    
    scheme = CompactFiniteDifferenceScheme()
    
    # 勾配計算
    gradient = scheme.compute_gradient(field, axis=0, boundary_condition=DummyBoundaryCondition())
    
    # ラプラシアン計算
    laplacian = scheme.compute_laplacian(field, boundary_condition=DummyBoundaryCondition())
    
    # 簡単な統計情報の表示
    print("元のフィールド:")
    print(f"  形状: {field.shape}")
    print(f"  最小値: {field.min():.4f}")
    print(f"  最大値: {field.max():.4f}")
    
    print("\n勾配:")
    print(f"  形状: {gradient.shape}")
    print(f"  最小値: {gradient.min():.4f}")
    print(f"  最大値: {gradient.max():.4f}")
    
    print("\nラプラシアン:")
    print(f"  形状: {laplacian.shape}")
    print(f"  最小値: {laplacian.min():.4f}")
    print(f"  最大値: {laplacian.max():.4f}")

# メイン実行
if __name__ == "__main__":
    demonstrate_compact_scheme()
