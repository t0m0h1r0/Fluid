# numerics/boundary.py
import numpy as np
from core.interfaces import BoundaryCondition, StencilOperator
from typing import Literal, Tuple, Optional, Union, List

class BaseBoundaryCondition(BoundaryCondition):
    """
    境界条件の基本実装
    
    境界条件の基本的な機能を提供する抽象基底クラス
    """
    def __init__(self, 
                 axis: int = 0, 
                 side: Literal['left', 'right'] = 'left'):
        """
        境界条件の初期化
        
        Args:
            axis: 境界条件を適用する軸
            side: 境界の側（左端または右端）
        """
        self.axis = axis
        self.side = side
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        境界条件の適用（基本実装）
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def get_stencil_operator(self) -> StencilOperator:
        """
        境界点での差分近似のための係数を返す
        
        Returns:
            ステンシル演算子
        """
        raise NotImplementedError("サブクラスで実装する必要があります")

class PeriodicBoundaryCondition(BaseBoundaryCondition):
    """
    周期境界条件の実装
    """
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        周期境界条件の適用
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        periodic_state = state.copy()
        
        # 前方境界の処理
        for i in range(2):
            idx_from = [slice(None)] * state.ndim
            idx_from[self.axis] = i
            idx_to = idx_from.copy()
            idx_to[self.axis] = -2 + i
            periodic_state[tuple(idx_from)] = state[tuple(idx_to)]
        
        # 後方境界の処理
        for i in range(2):
            idx_from = [slice(None)] * state.ndim
            idx_from[self.axis] = -(i + 1)
            idx_to = idx_from.copy()
            idx_to[self.axis] = 1 - i
            periodic_state[tuple(idx_from)] = state[tuple(idx_to)]
        
        return periodic_state
    
    def get_stencil_operator(self) -> StencilOperator:
        """
        周期境界条件のステンシル係数
        
        Returns:
            周期境界用のステンシル演算子
        """
        return StencilOperator(
            points=[-2, -1, 0, 1, 2],
            coefficients=[1/12, -2/3, 0, 2/3, -1/12]
        )

class NeumannBoundaryCondition(BaseBoundaryCondition):
    """
    ノイマン境界条件（勾配ゼロ）の実装
    """
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        ノイマン境界条件の適用
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        neumann_state = state.copy()
        
        # スライスの準備
        idx = [slice(None)] * state.ndim
        
        if self.side == 'left':
            # 左端の境界処理
            for i in range(2):
                idx[self.axis] = i
                next_idx = idx.copy()
                next_idx[self.axis] = i + 1
                neumann_state[tuple(idx)] = neumann_state[tuple(next_idx)]
        else:
            # 右端の境界処理
            for i in range(2):
                idx[self.axis] = -(i + 1)
                prev_idx = idx.copy()
                prev_idx[self.axis] = -(i + 2)
                neumann_state[tuple(idx)] = neumann_state[tuple(prev_idx)]
        
        return neumann_state
    
    def get_stencil_operator(self) -> StencilOperator:
        """
        ノイマン境界条件のステンシル係数
        
        Returns:
            ノイマン境界用のステンシル演算子
        """
        return StencilOperator(
            points=[0, 1, 2, 3, 4],
            coefficients=[-25/12, 4, -3, 4/3, -1/4]
        )

class DirichletBoundaryCondition(BaseBoundaryCondition):
    """
    ディリクレ境界条件の実装
    """
    def __init__(self, 
                 axis: int = 0, 
                 side: Literal['left', 'right'] = 'left',
                 value: Union[float, np.ndarray, None] = 0.0):
        """
        ディリクレ境界条件の初期化
        
        Args:
            axis: 境界条件を適用する軸
            side: 境界の側（左端または右端）
            value: 境界での固定値
        """
        super().__init__(axis, side)
        self.value = value
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        ディリクレ境界条件の適用
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        dirichlet_state = state.copy()
        
        # スライスの準備
        idx = [slice(None)] * state.ndim
        
        # 値の準備（スカラーまたは配列）
        if np.isscalar(self.value):
            boundary_value = self.value
        else:
            # 多次元配列の場合、適切な部分を選択
            boundary_slices = [slice(None)] * state.ndim
            boundary_slices[self.axis] = 0 if self.side == 'left' else -1
            boundary_value = self.value[tuple(boundary_slices)]
        
        if self.side == 'left':
            # 左端の境界処理
            idx[self.axis] = 0
            dirichlet_state[tuple(idx)] = boundary_value
            
            # 隣接点も同様の処理（必要に応じて）
            idx[self.axis] = 1
            dirichlet_state[tuple(idx)] = boundary_value
        else:
            # 右端の境界処理
            idx[self.axis] = -1
            dirichlet_state[tuple(idx)] = boundary_value
            
            # 隣接点も同様の処理（必要に応じて）
            idx[self.axis] = -2
            dirichlet_state[tuple(idx)] = boundary_value
        
        return dirichlet_state
    
    def get_stencil_operator(self) -> StencilOperator:
        """
        ディリクレ境界条件のステンシル係数
        
        Returns:
            ディリクレ境界用のステンシル演算子
        """
        return StencilOperator(
            points=[0],
            coefficients=[1.0]
        )

class MultiAxisBoundaryCondition(BaseBoundaryCondition):
    """
    多軸に対する境界条件の複合
    
    異なる軸に異なる境界条件を適用
    """
    def __init__(self, conditions: List[BoundaryCondition]):
        """
        多軸境界条件の初期化
        
        Args:
            conditions: 各軸の境界条件のリスト
        """
        self.conditions = conditions
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        多軸境界条件の適用
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        applied_state = state.copy()
        
        # 各軸に対して境界条件を順次適用
        for axis, condition in enumerate(self.conditions):
            # 軸に沿った部分配列を取得し、境界条件を適用
            slices = [slice(None)] * state.ndim
            slices[axis] = slice(None)
            
            # 部分配列に境界条件を適用
            applied_state[tuple(slices)] = condition.apply(
                applied_state[tuple(slices)]
            )
        
        return applied_state
    
    def get_stencil_operator(self) -> StencilOperator:
        """
        多軸境界条件のステンシル演算子
        
        最初の条件のステンシル演算子を返す
        
        Returns:
            最初の軸の境界条件のステンシル演算子
        """
        return self.conditions[0].get_stencil_operator()

# デモンストレーション関数
def demonstrate_boundary_conditions():
    """
    境界条件のデモンストレーション
    """
    # 3D配列の作成
    shape = (32, 32, 64)
    field = np.random.rand(*shape)
    
    # 単一軸の境界条件
    print("周期境界条件:")
    periodic_bc = PeriodicBoundaryCondition(axis=0)
    periodic_field = periodic_bc.apply(field)
    print(f"  元のフィールド: {field.shape}")
    print(f"  境界条件適用後: {periodic_field.shape}")
    
    print("\nノイマン境界条件:")
    neumann_bc = NeumannBoundaryCondition(axis=1, side='right')
    neumann_field = neumann_bc.apply(field)
    print(f"  元のフィールド: {field.shape}")
    print(f"  境界条件適用後: {neumann_field.shape}")
    
    print("\nディリクレ境界条件:")
    dirichlet_bc = DirichletBoundaryCondition(axis=2, side='left', value=0.5)
    dirichlet_field = dirichlet_bc.apply(field)
    print(f"  元のフィールド: {field.shape}")
    print(f"  境界条件適用後: {dirichlet_field.shape}")
    
    print("\n多軸境界条件:")
    multi_bc = MultiAxisBoundaryCondition([
        PeriodicBoundaryCondition(axis=0),
        NeumannBoundaryCondition(axis=1),
        DirichletBoundaryCondition(axis=2, value=0.0)
    ])
    multi_field = multi_bc.apply(field)
    print(f"  元のフィールド: {field.shape}")
    print(f"  境界条件適用後: {multi_field.shape}")

# メイン実行
if __name__ == "__main__":
    demonstrate_boundary_conditions()
