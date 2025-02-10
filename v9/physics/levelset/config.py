"""Level Set法の設定を管理するモジュール

このモジュールは、Level Set法の数値計算パラメータを定義・管理します。
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class LevelSetConfig:
    """Level Set法の数値計算パラメータ"""

    # 再初期化パラメータ
    reinitialize: Dict[str, Any] = field(
        default_factory=lambda: {
            "method": "fast_marching",
            "interval": 5,
            "steps": 2,
            "dt": 0.1,
        }
    )

    # 界面追跡パラメータ
    interface: Dict[str, Any] = field(
        default_factory=lambda: {
            "epsilon": 1.0e-2,  # 界面の厚さ
            "min_value": 1.0e-10,  # 最小値
        }
    )

    # 空間離散化パラメータ
    discretization: Dict[str, Any] = field(
        default_factory=lambda: {"scheme": "weno", "order": 5}
    )

    # 診断情報の保存設定
    diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {
            "compute_volume": True,
            "compute_area": True,
            "sample_points": None,
        }
    )

    def validate(self):
        """設定値の妥当性を検証"""
        # 再初期化パラメータの検証
        if not isinstance(self.reinitialize.get("interval", 5), int):
            raise ValueError("再初期化間隔は整数である必要があります")

        if not 0 < self.reinitialize.get("dt", 0.1) <= 1.0:
            raise ValueError("再初期化のdtは0より大きく1以下である必要があります")

        # 界面パラメータの検証
        if not 0 < self.interface.get("epsilon", 1e-2) < 1.0:
            raise ValueError("界面の厚さは0から1の間である必要があります")

        # 空間離散化の検証
        valid_schemes = ["weno", "central"]
        if self.discretization.get("scheme") not in valid_schemes:
            raise ValueError(f"無効な離散化スキーム。選択肢: {valid_schemes}")

        valid_weno_orders = [3, 5]
        if self.discretization.get("order") not in valid_weno_orders:
            raise ValueError(f"無効なWENO次数。選択肢: {valid_weno_orders}")

    def get_config_for_component(self, component: str) -> Dict[str, Any]:
        """特定のコンポーネントの設定を取得

        Args:
            component: 設定を取得するコンポーネント名

        Returns:
            コンポーネント固有の設定
        """
        component_configs = {
            "reinitialize": self.reinitialize,
            "interface": self.interface,
            "discretization": self.discretization,
            "diagnostics": self.diagnostics,
        }
        return component_configs.get(component, {})

    def save(self, filepath: str):
        """設定をYAMLファイルに保存

        Args:
            filepath: 保存先のパス
        """
        import yaml

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "reinitialize": self.reinitialize,
                    "interface": self.interface,
                    "discretization": self.discretization,
                    "diagnostics": self.diagnostics,
                },
                f,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(cls, filepath: str) -> "LevelSetConfig":
        """YAMLファイルから設定を読み込む

        Args:
            filepath: 読み込むYAMLファイルのパス

        Returns:
            読み込まれた設定インスタンス
        """
        import yaml

        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 設定の検証を含めた初期化
        config = cls(
            reinitialize=config_dict.get("reinitialize", {}),
            interface=config_dict.get("interface", {}),
            discretization=config_dict.get("discretization", {}),
            diagnostics=config_dict.get("diagnostics", {}),
        )
        config.validate()
        return config
