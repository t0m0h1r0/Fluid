"""2Dスカラー場の可視化を提供するモジュール"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Scalar2DRenderer:
    """2Dスカラー場のシンプルなレンダラー"""

    def __init__(self, config: Dict[str, Any] = None):
        """レンダラーを初期化

        Args:
            config: 可視化設定（オプション）
        """
        self.config = config or {}

    def render(
        self, 
        data: np.ndarray, 
        ax: Optional[Axes] = None, 
        **kwargs
    ) -> Tuple[Figure, Dict[str, Any]]:
        """2Dスカラー場を描画

        Args:
            data: スカラー場データ
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 入力バリデーション
        if data.ndim != 2:
            raise ValueError("2次元データが必要です")

        # 図とAxesの準備
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # デフォルトのパラメータ設定
        cmap = kwargs.get('cmap', 'viridis')
        interpolation = kwargs.get('interpolation', 'nearest')

        # メタデータの準備
        metadata = {
            'data_range': {
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data))
            },
            'display_type': ['scalar']
        }

        # スカラー場の描画
        im = ax.imshow(
            data.T, 
            origin='lower', 
            cmap=cmap,
            interpolation=interpolation,
            alpha=kwargs.get('alpha', 1.0)
        )

        # カラーバーの追加
        if kwargs.get('colorbar', True):
            plt.colorbar(im, ax=ax)

        # タイトルの追加
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

        # 軸ラベルの追加
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))

        return fig, metadata