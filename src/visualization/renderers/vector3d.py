"""3Dベクトル場の可視化を提供するモジュール"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
from matplotlib.axes import Axes


class Vector3DRenderer:
    """3Dベクトル場のシンプルなレンダラー"""

    def __init__(self, config: Dict[str, Any] = None):
        """レンダラーを初期化

        Args:
            config: 可視化設定（オプション）
        """
        self.config = config or {}

    def _get_slice(
        self, data: np.ndarray, slice_axis: str, slice_pos: float
    ) -> Tuple[np.ndarray, ...]:
        """指定された軸とスライス位置でデータをスライス

        Args:
            data: 入力データ配列
            slice_axis: スライス軸 ('xy', 'xz', 'yz')
            slice_pos: スライス位置 (0-1)

        Returns:
            スライスされたデータと座標
        """
        # データの形状を取得
        nx, ny, nz = data.shape

        # スライス位置のインデックスを計算
        if slice_axis == "xy":
            slice_idx = int(slice_pos * (nz - 1))
            slice_data = data[:, :, slice_idx]
            x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            z = np.full_like(x, slice_idx)
        elif slice_axis == "xz":
            slice_idx = int(slice_pos * (ny - 1))
            slice_data = data[:, slice_idx, :]
            x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
            y = np.full_like(x, slice_idx)
        elif slice_axis == "yz":
            slice_idx = int(slice_pos * (nx - 1))
            slice_data = data[slice_idx, :, :]
            y, z = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
            x = np.full_like(y, slice_idx)
        else:
            raise ValueError(f"無効なスライス軸: {slice_axis}")

        return slice_data, x, y, z

    def render(
        self,
        vector_components: List[np.ndarray],
        ax: Optional[Axes] = None,
        view: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """3Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v, w]
            ax: 既存のAxes（Noneの場合は新規作成）
            view: スライス情報
            **kwargs: 追加の描画オプション

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 入力バリデーション
        if len(vector_components) != 3:
            raise ValueError("3次元ベクトル場には3つの成分が必要です")

        u, v, w = vector_components

        # スライス情報の取得
        slice_axis = view.get("slice_axes", ["xy"])[0] if view else "xy"
        slice_pos = view.get("slice_positions", [0.5])[0] if view else 0.5

        # 図とAxesの準備
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # スライスの取得
        slice_data_list = []
        for component in [u, v, w]:
            slice_data, x, y, z = self._get_slice(component, slice_axis, slice_pos)
            slice_data_list.append(slice_data)

        # メタデータの準備
        metadata = {
            "data_range": {
                "min_magnitude": float(np.min(slice_data_list[0])),
                "max_magnitude": float(np.max(slice_data_list[0])),
            },
            "display_type": ["vector", "slice"],
            "slice_info": {"axis": slice_axis, "position": slice_pos},
        }

        # ベクトル場の描画
        ax.quiver(
            x,
            y,
            z,
            slice_data_list[0],
            slice_data_list[1],
            slice_data_list[2],
            color="blue",
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
