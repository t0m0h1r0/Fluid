# core/interfaces.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, TypeVar, Generic

# 型変数
T = TypeVar('T', bound=np.ndarray)

class PhysicalModel(ABC):
    """
    物理モデルの抽象基底クラス
    
    物理現象を記述する基本的な計算を定義
    """
    @abstractmethod
    def compute_flux(self, 
                     state: np.ndarray, 
                     parameters: Dict[str, Any]) -> np.ndarray:
        """
        状態から流束（フラックス）を計算
        
        Args:
            state: 現在の状態
            parameters: 計算に必要な追加パラメータ
        
        Returns:
            計算された流束
        """
        pass
    
    @abstractmethod
    def compute_source_terms(self, 
                             state: np.ndarray, 
                             parameters: Dict[str, Any]) -> np.ndarray:
        """
        ソース項（外部力、生成・消滅項など）の計算
        
        Args:
            state: 現在の状態
            parameters: 計算に必要な追加パラメータ
        
        Returns:
            ソース項
        """
        pass

class NumericalScheme(ABC):
    """
    数値スキームの抽象基底クラス
    
    離散化と数値的な近似計算を定義
    """
    @abstractmethod
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
        pass
    
    @abstractmethod
    def compute_gradient(self, 
                         field: np.ndarray, 
                         axis: int, 
                         boundary_condition: 'BoundaryCondition') -> np.ndarray:
        """
        勾配の計算
        
        Args:
            field: 入力フィールド
            axis: 勾配を計算する軸
            boundary_condition: 境界条件
        
        Returns:
            勾配場
        """
        pass
    
    @abstractmethod
    def compute_laplacian(self, 
                          field: np.ndarray, 
                          boundary_condition: 'BoundaryCondition') -> np.ndarray:
        """
        ラプラシアンの計算
        
        Args:
            field: 入力フィールド
            boundary_condition: 境界条件
        
        Returns:
            ラプラシアン
        """
        pass

class BoundaryCondition(ABC):
    """
    境界条件の抽象基底クラス
    
    境界での計算と条件の適用を定義
    """
    @abstractmethod
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        境界条件の適用
        
        Args:
            state: 入力状態
        
        Returns:
            境界条件を適用した状態
        """
        pass
    
    @abstractmethod
    def get_stencil_operator(self) -> 'StencilOperator':
        """
        境界点での差分近似のための係数を返す
        
        Returns:
            ステンシル演算子
        """
        pass

class StencilOperator:
    """
    差分スキームにおけるステンシル演算子
    
    境界点での差分近似のための係数を保持
    """
    def __init__(self, points: List[int], coefficients: List[float]):
        """
        ステンシル演算子の初期化
        
        Args:
            points: ステンシルの相対インデックス
            coefficients: 対応する係数
        """
        self.points = np.array(points)
        self.coefficients = np.array(coefficients)

class Solver(ABC, Generic[T]):
    """
    汎用ソルバーインターフェース
    
    状態の時間発展を解くための基本インターフェース
    """
    @abstractmethod
    def solve(self, 
              initial_state: T, 
              parameters: Dict[str, Any]) -> T:
        """
        状態の時間発展を解く
        
        Args:
            initial_state: 初期状態
            parameters: 追加パラメータ
        
        Returns:
            更新された状態
        """
        pass

class Field(Generic[T]):
    """
    汎用的な場（フィールド）クラス
    
    状態管理、保存則、時間発展を統合
    """
    def __init__(self, 
                 initial_state: T,
                 physical_model: PhysicalModel,
                 numerical_scheme: NumericalScheme,
                 boundary_condition: BoundaryCondition):
        """
        フィールドの初期化
        
        Args:
            initial_state: 初期状態
            physical_model: 物理モデル
            numerical_scheme: 数値スキーム
            boundary_condition: 境界条件
        """
        self._state = initial_state
        self._physical_model = physical_model
        self._numerical_scheme = numerical_scheme
        self._boundary_condition = boundary_condition
    
    def advance(self, dt: float, parameters: Optional[Dict[str, Any]] = None) -> T:
        """
        時間発展
        
        Args:
            dt: 時間刻み
            parameters: 追加パラメータ
        
        Returns:
            更新後の状態
        """
        # フラックスの計算
        flux = self._physical_model.compute_flux(
            self._state, 
            parameters or {}
        )
        
        # ソース項の計算
        source = self._physical_model.compute_source_terms(
            self._state,
            parameters or {}
        )
        
        # 数値スキームによる離散化
        new_state = self._numerical_scheme.discretize(
            flux, 
            source,
            self._state, 
            dt
        )
        
        # 境界条件の適用
        new_state = self._boundary_condition.apply(new_state)
        
        # 状態の更新
        self._state = new_state
        return self._state
    
    @property
    def state(self) -> T:
        """
        現在の状態への読み取り専用アクセス
        
        Returns:
            現在のフィールドの状態
        """
        return self._state

# プロファイリングと性能モニタリングのためのデコレータ
def profile_method(func):
    """
    メソッドの実行時間と呼び出し回数を計測するデコレータ
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # プロファイリング情報の記録（実際のアプリケーションでは別の方法で管理）
        print(f"Method {func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper

# デバッグと検証のためのユーティリティ関数
def validate_array(arr: np.ndarray, 
                   name: str = 'Array', 
                   check_finite: bool = True,
                   min_val: Optional[float] = None,
                   max_val: Optional[float] = None):
    """
    配列の検証を行うユーティリティ関数
    
    Args:
        arr: 検証する配列
        name: 配列の名前（デバッグ出力用）
        check_finite: 有限値のみかをチェック
        min_val: 最小値の制限
        max_val: 最大値の制限
    
    Raises:
        ValueError: 配列が検証基準を満たさない場合
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    
    if check_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    
    if min_val is not None and np.any(arr < min_val):
        raise ValueError(f"{name} contains values below {min_val}")
    
    if max_val is not None and np.any(arr > max_val):
        raise ValueError(f"{name} contains values above {max_val}")
