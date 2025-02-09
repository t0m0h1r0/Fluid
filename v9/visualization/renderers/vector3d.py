"""3Dベクトル場の可視化を提供するモジュール"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


class Vector3DRenderer:
    """3Dベクトル場のシンプルなレンダラー"""

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
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """3Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v, w]
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 入力バリデーション
        if len(vector_components) != 3:
            raise ValueError("3次元ベクトル場には3つの成分が必要です")
        
        u, v, w = vector_components

        # 図とAxesの準備
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # ベクトルの大きさを計算
        magnitude = np.sqrt(u**2 + v**2 + w**2)

        # デフォルトのパラメータ設定
        density = kwargs.get('density', 10)
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
        nx, ny, nz = u.shape
        x, y, z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )

        # ベクトル場の描画
        ax.quiver(
            x[::skip, ::skip, ::skip], 
            y[::skip, ::skip, ::skip], 
            z[::skip, ::skip, ::skip],
            u[::skip, ::skip, ::skip], 
            v[::skip, ::skip, ::skip], 
            w[::skip, ::skip, ::skip],
            length=scale,
            normalize=True,
            color='blue',  # 単純な静的な色
            alpha=kwargs.get('alpha', 0.7)
        )

        # タイトルの追加
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

        # 軸ラベルの設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return fig, metadata