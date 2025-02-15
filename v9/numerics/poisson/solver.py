"""
Poisson方程式ソルバーの設定管理モジュール

このモジュールは、数値偏微分方程式の解法に関する設定を
柔軟かつ堅牢に管理するための機能を提供します。

主な特徴:
1. 反復法のパラメータ設定
2. 収束判定基準の定義
3. デフォルト値と設定のバリデーション
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import yaml


@dataclass
class PoissonSolverConfig:
    """
    Poisson方程式ソルバーの設定を管理するデータクラス

    数値解法の主要パラメータを包括的に管理します。

    主要な設定パラメータ:
    - max_iterations: 最大反復回数
    - tolerance: 収束判定の許容誤差
    - absolute_tolerance: 絶対誤差による収束判定
    - relaxation_parameter: 緩和パラメータ（SOR法など）
    """

    max_iterations: int = 1000
    tolerance: float = 1e-6
    absolute_tolerance: bool = False
    relaxation_parameter: float = 1.0

    # オプションの診断設定
    diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {"save_residual_history": True, "log_frequency": 10}
    )

    def validate(self) -> None:
        """
        設定値の妥当性を検証

        数値的制約と論理的整合性を確認します。
        """
        if self.max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")

        if self.tolerance <= 0:
            raise ValueError("収束判定の許容誤差は正の値である必要があります")

        if not 0 < self.relaxation_parameter <= 2:
            raise ValueError("緩和パラメータは0から2の間である必要があります")

    def update(self, config_dict: Dict[str, Any]) -> "PoissonSolverConfig":
        """
        設定を更新し、新しい設定インスタンスを返します。

        Args:
            config_dict: 更新する設定の辞書

        Returns:
            更新された設定インスタンス
        """
        updated_config = PoissonSolverConfig(**{**self.__dict__, **config_dict})
        updated_config.validate()
        return updated_config

    @classmethod
    def from_yaml(cls, filepath: str) -> "PoissonSolverConfig":
        """
        YAMLファイルから設定を読み込みます。

        Args:
            filepath: 設定ファイルのパス

        Returns:
            読み込まれた設定インスタンス
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        config = cls(**config_dict)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        設定を辞書形式にシリアライズします。

        Returns:
            設定の辞書表現
        """
        return {
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "relaxation_parameter": self.relaxation_parameter,
            "diagnostics": self.diagnostics,
        }

    def save_to_yaml(self, filepath: str) -> None:
        """
        設定をYAMLファイルに保存します。

        Args:
            filepath: 保存先のファイルパス
        """
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
