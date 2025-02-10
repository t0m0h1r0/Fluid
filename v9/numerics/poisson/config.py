"""Poisson方程式ソルバーの設定を管理するモジュール

このモジュールは、Poisson方程式の数値計算パラメータを定義・管理します。
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import yaml


@dataclass
class PoissonSolverConfig:
    """Poisson方程式ソルバーの数値計算パラメータ"""

    # 収束判定パラメータ
    convergence: Dict[str, Any] = field(
        default_factory=lambda: {
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "relative_tolerance": False,
        }
    )

    # ソルバー固有のパラメータ
    solver_specific: Dict[str, Any] = field(
        default_factory=lambda: {
            "relaxation_parameter": 1.5,  # SORなどで使用
            "auto_tune": False,
            "method": "sor",
        }
    )

    # 診断情報の保存設定
    diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {"save_residual_history": True, "log_frequency": 10}
    )

    def validate(self):
        """設定値の妥当性を検証"""
        # 収束判定パラメータの検証
        if not 0 < self.convergence.get("tolerance", 1e-6) < 1:
            raise ValueError("許容誤差は0から1の間である必要があります")

        if not isinstance(self.convergence.get("max_iterations", 1000), int):
            raise ValueError("最大反復回数は整数である必要があります")

        # ソルバー固有のパラメータ検証
        if not 0 < self.solver_specific.get("relaxation_parameter", 1.5) <= 2:
            raise ValueError("緩和パラメータは0から2の間である必要があります")

        # 有効なソルバー方法のチェック
        valid_methods = ["sor", "jacobi", "gauss_seidel"]
        if self.solver_specific.get("method") not in valid_methods:
            raise ValueError(f"無効なソルバー方法。選択肢: {valid_methods}")

    def get_config_for_component(self, component: str) -> Dict[str, Any]:
        """特定のコンポーネントの設定を取得

        Args:
            component: 設定を取得するコンポーネント名

        Returns:
            コンポーネント固有の設定
        """
        component_configs = {
            "convergence": self.convergence,
            "solver_specific": self.solver_specific,
            "diagnostics": self.diagnostics,
        }
        return component_configs.get(component, {})

    def save(self, filepath: str):
        """設定をYAMLファイルに保存

        Args:
            filepath: 保存先のパス
        """
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "convergence": self.convergence,
                    "solver_specific": self.solver_specific,
                    "diagnostics": self.diagnostics,
                },
                f,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(cls, filepath: str) -> "PoissonSolverConfig":
        """YAMLファイルから設定を読み込む

        Args:
            filepath: 読み込むYAMLファイルのパス

        Returns:
            読み込まれた設定インスタンス
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 設定の検証を含めた初期化
        config = cls(
            convergence=config_dict.get("convergence", {}),
            solver_specific=config_dict.get("solver_specific", {}),
            diagnostics=config_dict.get("diagnostics", {}),
        )
        config.validate()
        return config
