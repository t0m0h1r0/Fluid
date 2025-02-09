"""2Dベクトル場の可視化を提供するモジュール"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Vector2DRenderer:
    """2Dベクトル場のシンプルなレンダラー"""

    def __init__(self, config: Dict[str, Any] = None):
        """レンダラーを初期化

        Args:
            config: 可視化設定（オプション）
        """
        self.config = config or {}

    def render(
        self, 
        vector_components: List[np.ndarray], 
        ax: Optional[Axes] = None, 
        **kwargs
    ) -> Tuple[Figure, Dict[str, Any]]:
        """2Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v]
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 入力バリデーション
        if len(vector_components) != 2:
            raise ValueError("2次元ベクトル場には2つの成分が必要です")
        
        u, v = vector_components

        # 図とAxesの準備
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # ベクトルの大きさを計算
        magnitude = np.sqrt(u**2 + v**2)

        # デフォルトのパラメータ設定
        density = kwargs.get('density', 20)
        scale = kwargs.get('scale', 1.0)
        skip = max(1, min(u.shape) // density)

        # メタデータの準備
        metadata = {
            'data_range': {
                'min_magnitude': float(np.min(magnitude)),
                'max_magnitude': float(np.max(magnitude))
            },
            'display_type': ['vector']
        }

        # グリッドの生成
        nx, ny = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # ベクトル場の描画
        ax.quiver(
            X[::skip, ::skip], 
            Y[::skip, ::skip], 
            u[::skip, ::skip], 
            v[::skip, ::skip],
            color='blue',  # 単純な静的な色
            scale=scale,
            alpha=kwargs.get('alpha', 0.7)
        )

        # タイトルの追加
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

        # アスペクト比と軸ラベル
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        return fig, metadata