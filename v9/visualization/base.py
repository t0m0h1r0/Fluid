"""可視化システムの基底クラスを提供するモジュール

このモジュールは、2D/3D可視化の基底となるクラスを定義し、
共通のインターフェースと基本機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """可視化の設定
    
    Attributes:
        output_dir: 出力ディレクトリ
        image_format: 画像フォーマット（png, jpg, etc.）
        dpi: 解像度（DPI）
        colormap: カラーマップ
        background_color: 背景色
        show_axes: 軸の表示
        show_grid: グリッドの表示
        show_colorbar: カラーバーの表示
    """
    output_dir: str = "visualization"
    image_format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    background_color: str = "white"
    show_axes: bool = True
    show_grid: bool = True
    show_colorbar: bool = True

class Visualizer(ABC):
    """可視化システムの基底クラス
    
    2D/3D可視化に共通のインターフェースを定義します。
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """可視化システムを初期化
        
        Args:
            config: 可視化の設定
        """
        self.config = config or VisualizationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def visualize_scalar(self, data: np.ndarray, name: str,
                        timestamp: float, **kwargs) -> None:
        """スカラー場を可視化
        
        Args:
            data: スカラー場のデータ
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
        """
        pass
    
    @abstractmethod
    def visualize_vector(self, data: List[np.ndarray], name: str,
                        timestamp: float, **kwargs) -> None:
        """ベクトル場を可視化
        
        Args:
            data: ベクトル場の各成分のデータ
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
        """
        pass
    
    @abstractmethod
    def visualize_interface(self, levelset: np.ndarray,
                          timestamp: float, **kwargs) -> None:
        """界面を可視化
        
        Args:
            levelset: Level Set関数のデータ
            timestamp: 時刻
            **kwargs: 追加の設定
        """
        pass
    
    def create_filename(self, name: str, timestamp: float) -> str:
        """出力ファイル名を生成
        
        Args:
            name: 変数名
            timestamp: 時刻
            
        Returns:
            生成されたファイル名
        """
        return str(self.output_dir / 
                  f"{name}_{timestamp:.6f}.{self.config.image_format}")
    
    def get_scale_and_range(self, data: np.ndarray,
                           symmetric: bool = False) -> Tuple[float, Tuple[float, float]]:
        """データのスケールと範囲を計算
        
        Args:
            data: 入力データ
            symmetric: 対称な範囲にするかどうか
            
        Returns:
            (スケール, (最小値, 最大値))のタプル
        """
        data_min = np.min(data)
        data_max = np.max(data)
        
        if symmetric:
            abs_max = max(abs(data_min), abs(data_max))
            scale = abs_max
            value_range = (-abs_max, abs_max)
        else:
            scale = data_max - data_min
            value_range = (data_min, data_max)
        
        return scale, value_range
    
    def add_colorbar(self, mappable: Any, label: str,
                    orientation: str = 'vertical') -> None:
        """カラーバーを追加
        
        Args:
            mappable: カラーマップを持つオブジェクト
            label: カラーバーのラベル
            orientation: カラーバーの向き
        """
        pass  # 各実装で必要に応じてオーバーライド
    
    def apply_common_settings(self, ax: Any) -> None:
        """共通の描画設定を適用
        
        Args:
            ax: プロット用のAxes
        """
        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(self.config.background_color)
        
        if self.config.show_axes:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            if hasattr(ax, 'set_zlabel'):
                ax.set_zlabel('Z')
        else:
            ax.set_axis_off()
        
        if self.config.show_grid and hasattr(ax, 'grid'):
            ax.grid(True)
    
    def create_figure(self, size: Tuple[float, float] = (8, 6),
                     projection: Optional[str] = None) -> Tuple[Any, Any]:
        """図とAxesを作成
        
        Args:
            size: 図のサイズ
            projection: プロジェクションの種類
            
        Returns:
            (図, Axes)のタプル
        """
        pass  # 各実装で必要に応じてオーバーライド