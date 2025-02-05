# parallel_utils.py
import multiprocessing
import numpy as np
from numba import njit, prange # type: ignore
from functools import partial
from typing import Callable, Any, Tuple

class ParallelCompute:
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def parallel_gradient(field: np.ndarray, axis: int, spacing: float = 1.0) -> np.ndarray:
        """並列化された勾配計算
        
        Args:
            field: 入力スカラー場
            axis: 微分を計算する軸
            spacing: グリッド間隔
        
        Returns:
            勾配場
        """
        gradient = np.zeros_like(field)
        shape = field.shape
        ndim = field.ndim
        
        # より効率的なインデックス計算
        dims = list(range(ndim))
        dims.remove(axis)
        
        # 内側ループの最適化
        sizes = [shape[d] for d in dims]
        total_size = np.prod(sizes)
        
        @njit
        def compute_slice(idx: int) -> Tuple[int, ...]:
            """多次元インデックスの計算を最適化"""
            indices = []
            temp = idx
            for size in sizes[:-1]:
                indices.append(temp // size)
                temp = temp % size
            indices.append(temp)
            return tuple(indices)
        
        # メインの並列ループ
        for idx in prange(total_size):
            # 多次元インデックスの変換を最適化
            multi_idx = compute_slice(idx)
            
            # スライスの作成を最適化
            idx_full = list(multi_idx)
            idx_full.insert(axis, slice(None))
            line = field[tuple(idx_full)]
            
            # 勾配計算の最適化（4次精度中心差分）
            grad_line = np.zeros_like(line)
            
            # 内部点での4次精度中心差分
            n = len(line)
            for i in range(2, n-2):
                grad_line[i] = (-line[i+2] + 8*line[i+1] - 8*line[i-1] + line[i-2]) / (12 * spacing)
            
            # 境界での4次精度片側差分
            # 前方境界
            grad_line[0] = (-25*line[0] + 48*line[1] - 36*line[2] + 
                          16*line[3] - 3*line[4]) / (12 * spacing)
            grad_line[1] = (-3*line[0] - 10*line[1] + 18*line[2] - 
                          6*line[3] + line[4]) / (12 * spacing)
            
            # 後方境界
            grad_line[-1] = (25*line[-1] - 48*line[-2] + 36*line[-3] -
                           16*line[-4] + 3*line[-5]) / (12 * spacing)
            grad_line[-2] = (3*line[-1] + 10*line[-2] - 18*line[-3] + 
                           6*line[-4] - line[-5]) / (12 * spacing)
            
            # 結果の格納を最適化
            idx_full_grad = list(multi_idx)
            idx_full_grad.insert(axis, slice(None))
            gradient[tuple(idx_full_grad)] = grad_line
        
        return gradient

    @staticmethod
    def parallel_compute_field(func: Callable[[np.ndarray], np.ndarray], 
                             field: np.ndarray,
                             chunk_size: int = None,
                             **kwargs) -> np.ndarray:
        """指定された関数を並列実行
        
        Args:
            func: 並列実行する関数
            field: 入力フィールド
            chunk_size: チャンクサイズ（Noneの場合は自動設定）
            **kwargs: 追加のキーワード引数
            
        Returns:
            処理後のフィールド
        """
        # CPUコア数とチャンクサイズの最適化
        num_cores = multiprocessing.cpu_count()
        if chunk_size is None:
            chunk_size = max(1, field.shape[0] // (num_cores * 4))
        
        # フィールドの分割を最適化
        chunks = []
        for i in range(0, field.shape[0], chunk_size):
            end = min(i + chunk_size, field.shape[0])
            chunks.append(field[i:end])
        
        # 並列処理の実行
        with multiprocessing.Pool(processes=num_cores) as pool:
            processed_chunks = pool.map(
                partial(func, **kwargs),
                chunks
            )
        
        # 結果の結合を最適化
        return np.concatenate(processed_chunks)

    @staticmethod
    def create_parallel_operator(func: Callable,
                               chunk_size: int = None,
                               **kwargs) -> Callable:
        """任意の関数を並列化可能な演算子に変換
        
        Args:
            func: 並列化する関数
            chunk_size: チャンクサイズ（Noneの場合は自動設定）
            **kwargs: 追加のキーワード引数
            
        Returns:
            並列化された関数
        """
        if not callable(func):
            raise ValueError("引数は呼び出し可能な関数である必要があります。")
        
        def parallel_operator(field: np.ndarray) -> np.ndarray:
            return ParallelCompute.parallel_compute_field(
                partial(func, **kwargs),
                field,
                chunk_size=chunk_size
            )
        
        return parallel_operator

    @staticmethod
    @njit
    def static_gradient(field: np.ndarray, axis: int, spacing: float = 1.0) -> np.ndarray:
        """JITコンパイルされた非並列勾配計算（小規模問題用）"""
        gradient = np.zeros_like(field)
        shape = field.shape
        
        # 内部点の計算を最適化（4次精度中心差分）
        slices = [slice(None)] * field.ndim
        slices[axis] = slice(2, -2)
        
        # メインの計算部分
        for idx in np.ndindex(tuple(s.stop-s.start if isinstance(s, slice) else s 
                                  for i, s in enumerate(slices))):
            # インデックスの設定
            full_idx = list(idx)
            axis_idx = full_idx[axis] + 2  # スライスのオフセットを考慮
            
            # 4次精度中心差分の計算
            p2 = field[tuple(slice_at(slices, axis, axis_idx+2))]
            p1 = field[tuple(slice_at(slices, axis, axis_idx+1))]
            m1 = field[tuple(slice_at(slices, axis, axis_idx-1))]
            m2 = field[tuple(slice_at(slices, axis, axis_idx-2))]
            
            gradient[tuple(slice_at(slices, axis, axis_idx))] = \
                (-p2 + 8*p1 - 8*m1 + m2) / (12 * spacing)
        
        # 境界の処理（4次精度片側差分）
        # 前方境界
        slices[axis] = 0
        gradient[tuple(slices)] = compute_forward_boundary(
            field, axis, slices, spacing)
        
        # 後方境界
        slices[axis] = -1
        gradient[tuple(slices)] = compute_backward_boundary(
            field, axis, slices, spacing)
        
        return gradient

@njit
def slice_at(slices: list, axis: int, index: int) -> tuple:
    """特定のインデックスでのスライス計算を最適化"""
    new_slices = slices.copy()
    new_slices[axis] = index
    return tuple(new_slices)

@njit
def compute_forward_boundary(field: np.ndarray, 
                           axis: int, 
                           slices: list, 
                           spacing: float) -> np.ndarray:
    """前方境界での4次精度片側差分"""
    idx = list(slices)
    idx[axis] = slice(None, 5)
    values = field[tuple(idx)]
    
    return (-25*values[0] + 48*values[1] - 36*values[2] + 
            16*values[3] - 3*values[4]) / (12 * spacing)

@njit
def compute_backward_boundary(field: np.ndarray, 
                            axis: int, 
                            slices: list, 
                            spacing: float) -> np.ndarray:
    """後方境界での4次精度片側差分"""
    idx = list(slices)
    idx[axis] = slice(-5, None)
    values = field[tuple(idx)]
    
    return (25*values[-1] - 48*values[-2] + 36*values[-3] - 
            16*values[-4] + 3*values[-5]) / (12 * spacing)