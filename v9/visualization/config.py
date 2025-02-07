"""可視化システムの設定を管理するモジュール

設定の読み込みと管理を行います。
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import yaml
import os


@dataclass
class VisualizationConfig:
    """可視化設定を管理するクラス

    YAMLファイルから読み込む可視化パラメータを定義します。
    """
    output_dir: str = "results/visualization"
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False
    
    vector_plot: Dict[str, Any] = field(default_factory=lambda: {
        "density": 20,
        "scale": 1.0
    })
    
    scalar_plot: Dict[str, Any] = field(default_factory=lambda: {
        "interpolation": "nearest",
        "contour": False
    })
    
    interface_plot: Dict[str, Any] = field(default_factory=lambda: {
        "filled": True,
        "colors": ["lightblue", "white"]
    })

    @classmethod
    def from_yaml(cls, config_path_or_dict):
        """YAMLファイルまたは設定辞書から設定を読み込む

        Args:
            config_path_or_dict: YAMLファイルのパスまたは設定辞書

        Returns:
            VisualizationConfigインスタンス
        """
        # 辞書が渡された場合
        if isinstance(config_path_or_dict, dict):
            viz_config = config_path_or_dict.get('visualization', {})
        else:
            # ファイルパスが渡された場合
            with open(config_path_or_dict, 'r') as f:
                config = yaml.safe_load(f)
            viz_config = config.get('visualization', {})

        return cls(
            output_dir=viz_config.get('output_dir', 'results/visualization'),
            format=viz_config.get('format', 'png'),
            dpi=viz_config.get('dpi', 300),
            colormap=viz_config.get('colormap', 'viridis'),
            show_colorbar=viz_config.get('show_colorbar', True),
            show_axes=viz_config.get('show_axes', True),
            show_grid=viz_config.get('show_grid', False),
            vector_plot=viz_config.get('vector_plot', {}),
            scalar_plot=viz_config.get('scalar_plot', {}),
            interface_plot=viz_config.get('interface_plot', {})
        )

    def ensure_output_dir(self):
        """出力ディレクトリを作成"""
        os.makedirs(self.output_dir, exist_ok=True)

    def get_output_filename(self, prefix: str, timestamp: float) -> str:
        """出力ファイル名を生成

        Args:
            prefix: ファイル名のプレフィックス
            timestamp: タイムスタンプ

        Returns:
            生成されたファイルパス
        """
        filename = f"{prefix}_{timestamp:.6f}.{self.format}"
        return os.path.join(self.output_dir, filename)