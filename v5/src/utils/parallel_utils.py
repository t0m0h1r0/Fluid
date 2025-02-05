import multiprocessing
import numpy as np
from numba import njit, prange
from functools import partial

class ParallelCompute:
    @staticmethod
    @njit(parallel=True)
    def parallel_gradient(field: np.ndarray, axis: int, spacing: float = 1.0) -> np.ndarray:
        """
        並列化された勾配計算
        
        Args:
            field (np.ndarray): 入力スカラー場
            axis (int): 微分を計算する軸
            spacing (float): グリッド間隔
        
        Returns:
            np.ndarray: 勾配場
        """
        gradient = np.zeros_like(field)
        shape = field.shape
        
        # 軸に垂直な方向のインデックスを生成
        other_axes = tuple(ax for ax in range(field.ndim) if ax != axis)
        
        for idx in prange(np.prod(np.array(shape)[list(other_axes)])):
            # 多次元インデックスに変換
            multi_idx = np.unravel_index(idx, np.array(shape)[list(other_axes)])
            
            # スライスを作成
            idx_full = list(multi_idx)
            idx_full.insert(axis, slice(None))
            line = field[tuple(idx_full)]
            
            # 中心差分
            grad_line = np.zeros_like(line)
            grad_line[1:-1] = (line[2:] - line[:-2]) / (2 * spacing)
            
            # 端点の処理（1次精度片側差分）
            grad_line[0] = (line[1] - line[0]) / spacing
            grad_line[-1] = (line[-1] - line[-2]) / spacing
            
            # 結果を戻す
            idx_full_grad = list(multi_idx)
            idx_full_grad.insert(axis, slice(None))
            gradient[tuple(idx_full_grad)] = grad_line
        
        return gradient
    
    @staticmethod
    def parallel_compute_field(func, field: np.ndarray, *args, **kwargs):
        """
        指定された関数を並列実行
        
        Args:
            func (callable): 並列実行する関数
            field (np.ndarray): 入力フィールド
            *args: 追加の位置引数
            **kwargs: 追加のキーワード引数
        
        Returns:
            np.ndarray: 処理後のフィールド
        """
        # CPUコア数を取得
        num_cores = multiprocessing.cpu_count()
        
        # フィールドを分割
        split_fields = np.array_split(field, num_cores)
        
        # 並列処理
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(
                func, 
                [(f, *args) for f in split_fields]
            )
        
        # 結果を結合
        return np.concatenate(results)

    @staticmethod
    def create_parallel_operator(func):
        """
        任意の関数を並列化可能な演算子に変換
        
        Args:
            func (callable): 並列化する関数
        
        Returns:
            callable: 並列化された関数
        """
        def parallel_operator(field, *args, **kwargs):
            # funcが完全な関数でない場合、partialを使用
            if not hasattr(func, '__call__'):
                raise ValueError("引数は呼び出し可能な関数である必要があります。")
            
            # パラレルコンピュートを使用
            return ParallelCompute.parallel_compute_field(
                # funcが引数を受け取れるようにpartialを使用
                partial(func, *args, **kwargs), 
                field
            )
        return parallel_operator

    @staticmethod
    def static_gradient(field: np.ndarray, axis: int, spacing: float = 1.0) -> np.ndarray:
        """
        NumbaのJITコンパイルを使用しない通常の勾配計算
        """
        gradient = np.zeros_like(field)
        
        # 多次元対応の勾配計算
        for idx in np.ndindex(field.shape[:axis] + field.shape[axis+1:]):
            # インデックス生成
            idx_list = list(idx)
            idx_list.insert(axis, slice(None))
            line = field[tuple(idx_list)]
            
            # 中心差分
            grad_line = np.zeros_like(line)
            grad_line[1:-1] = (line[2:] - line[:-2]) / (2 * spacing)
            
            # 端点の処理（1次精度片側差分）
            grad_line[0] = (line[1] - line[0]) / spacing
            grad_line[-1] = (line[-1] - line[-2]) / spacing
            
            # 結果を戻す
            idx_list_grad = list(idx)
            idx_list_grad.insert(axis, slice(None))
            gradient[tuple(idx_list_grad)] = grad_line
        
        return gradient