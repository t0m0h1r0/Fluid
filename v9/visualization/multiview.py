"""可視化システムの多視点可視化モジュール

このモジュールは、シミュレーション状態の多視点可視化を提供します。
"""

from pathlib import Path
from typing import List

from .core.base import VisualizationConfig, ViewConfig
from .interfaces import VisualizationContext, VisualizationFactory


def create_multiview_visualization(
    state,
    config: VisualizationConfig,
    timestamp: float = 0.0,
    base_name: str = "simulation_state",
) -> List[str]:
    """シミュレーション状態の多視点可視化を生成"""
    # 出力ディレクトリの作成
    if hasattr(config, "output_dir"):
        output_dir = Path(config.output_dir)
    elif isinstance(config, dict):
        output_dir = Path(config.get("output_dir", "results/visualization"))
    else:
        output_dir = Path("results/visualization")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    # 可視化戦略の選択（次元数に基づく）
    ndim = len(state.velocity.components[0].data.shape)
    strategy_type = "3d" if ndim == 3 else "2d"
    viz_context = VisualizationContext(
        VisualizationFactory.create_strategy(strategy_type, config)
    )

    # 可視化設定とフィールド設定の取得を修正
    if isinstance(config, dict):
        viz_config = config.get("visualization", {})
        fields_config = viz_config.get("fields", {})
        slice_config = viz_config.get("slices", {})
    elif hasattr(config, "output"):
        viz_config = config.output
        fields_config = viz_config.fields
        slice_config = getattr(viz_config, "slices", {})
    else:
        viz_config = {}
        fields_config = {}
        slice_config = {}

    # スライス設定の取得
    slice_axes = slice_config.get("axes", ["xy"])
    slice_positions = slice_config.get("positions", [0.5])

    # 可視化する物理量の設定
    physics_fields = []

    # 速度場の設定
    if fields_config.get("velocity", {}).get("enabled", True):
        physics_fields.append(
            {
                "name": "velocity",
                "data": [comp.data for comp in state.velocity.components],
                "plot_type": "vector",
            }
        )

    # 圧力場の設定
    if fields_config.get("pressure", {}).get("enabled", True):
        physics_fields.append(
            {"name": "pressure", "data": state.pressure.data, "plot_type": "scalar"}
        )

    # レベルセット場の設定
    if fields_config.get("levelset", {}).get("enabled", True):
        physics_fields.append(
            {"name": "levelset", "data": state.levelset.data, "plot_type": "scalar"}
        )

    # physics_fieldsが空の場合のデフォルト設定
    if not physics_fields:
        physics_fields = [
            {
                "name": "velocity",
                "data": [comp.data for comp in state.velocity.components],
                "plot_type": "vector",
            },
            {"name": "pressure", "data": state.pressure.data, "plot_type": "scalar"},
            {"name": "levelset", "data": state.levelset.data, "plot_type": "scalar"},
        ]

    # 可視化の実行
    for field in physics_fields:
        for slice_axis in slice_axes:
            for slice_pos in slice_positions:
                try:
                    # ViewConfigの作成
                    view_config = ViewConfig(
                        slice_axes=[slice_axis], slice_positions=[slice_pos]
                    )

                    # ファイル名の生成
                    filename = (
                        f"{base_name}_{field['name']}"
                        f"_{slice_axis}_slice_{slice_pos:.2f}_{timestamp:.3f}"
                    )

                    # 可視化の実行
                    filepath = viz_context.visualize(
                        field["data"],
                        name=filename,
                        timestamp=timestamp,
                        view=view_config,
                    )

                    output_files.append(filepath)

                except Exception as e:
                    print(f"可視化中にエラー発生: {e}")
                    import traceback

                    traceback.print_exc()

    return output_files


def visualize_simulation_state(state, config, timestamp: float = 0.0) -> List[str]:
    """シミュレーション状態を可視化

    Args:
        state: シミュレーション状態
        config: 可視化設定
        timestamp: 現在の時刻

    Returns:
        生成された可視化ファイルのパス一覧
    """
    # 設定が辞書の場合、VisualizationConfigに変換
    if isinstance(config, dict):
        config = VisualizationConfig.from_dict(config)

    return create_multiview_visualization(state, config, timestamp=timestamp)
