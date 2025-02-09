"""3Dスカラー場の可視化を提供するモジュール"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Union
from matplotlib.axes import Axes


class Scalar3DRenderer:
    """3Dスカラー場のシンプルなレンダラー"""

    def __init__(self, config: Dict[str, Any] = None):
        """レンダラーを初期化

        Args:
            config: 可視化設定（オプション）
        """
        self.config = config or {}

    def render(
        self,
        data: Union[np.ndarray, int],
        ax: Optional[Axes] = None,
        view: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """3Dスカラー場を描画

        Args:
            data: スカラー場データ
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 入力データのチェック
        if isinstance(data, int):
            raise ValueError("データは数値ではなく、numpy配列である必要があります")

        # 入力バリデーション
        if data.ndim != 3:
            raise ValueError("3次元データが必要です")

        # 図とAxesの準備
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # スライス位置の決定
        _ = {"yz": 0, "xz": 1, "xy": 2}
        if view is None:
            slice_axis = 2
            slice_pos = data.shape[0] // 2
        else:
            slice_axis = _[view.slice_axes[0]]
            slice_pos = int(view.slice_positions[0] * data.shape[0])

        # スライスの抽出
        slices = [slice(None)] * 3
        slices[slice_axis] = slice_pos
        slice_data = data[tuple(slices)].T

        # メタデータの準備
        metadata = {
            "data_range": {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
            },
            "display_type": ["scalar", "slice"],
        }

        # グリッドの生成
        nx, ny = slice_data.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        # 座標の調整
        if slice_axis == 0:
            X, Y, Z = slice_pos * np.ones_like(X), X, Y
        elif slice_axis == 1:
            X, Y, Z = X, slice_pos * np.ones_like(X), Y
        else:
            X, Y, Z = X, Y, slice_pos * np.ones_like(X)

        # カラーマップと色の設定
        cmap = kwargs.get("cmap", "viridis")

        # データの正規化
        normalized_data = (slice_data - slice_data.min()) / (
            slice_data.max() - slice_data.min()
        )

        im = ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=plt.cm.get_cmap(cmap)(normalized_data),
            alpha=kwargs.get("alpha", 0.7),
        )

        # タイトルの追加
        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        # 軸ラベルの設定
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        return fig, metadata
