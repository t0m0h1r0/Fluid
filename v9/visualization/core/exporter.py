"""可視化結果のエクスポートを提供するモジュール

このモジュールは、Matplotlibの描画結果を画像ファイルとして
出力する機能を実装します。
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .base import Exporter, VisualizationConfig


class ImageExporter(Exporter):
    """画像エクスポーター

    Matplotlibの描画結果を画像ファイルとして出力します。
    メタデータもJSONファイルとして保存できます。
    """

    def __init__(self, config: VisualizationConfig):
        """画像エクスポーターを初期化"""
        super().__init__(config)

    def export(
        self,
        figure: plt.Figure,
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """描画結果を画像として出力

        Args:
            figure: Matplotlibの図
            filepath: 出力ファイルパス
            metadata: メタデータ（オプション）
        """
        # 出力ディレクトリの作成
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 画像として保存
        figure.savefig(
            filepath,
            dpi=self.config.dpi,
            format=self.config.format,
            bbox_inches="tight",
        )

        # メタデータの保存
        if metadata is not None:
            metadata_path = filepath.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 図のクリーンアップ
        plt.close(figure)
