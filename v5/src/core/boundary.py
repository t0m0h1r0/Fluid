# boundary.py
import numpy as np
from .scheme import BoundaryCondition, StencilOperator


class PeriodicBC(BoundaryCondition):
    def get_stencil_operator(self) -> StencilOperator:
        """4次精度中心差分の係数を返す"""
        return StencilOperator(
            points=[-2, -1, 0, 1, 2], coefficients=[1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
        )

    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        """多次元周期境界条件の実装"""
        # コピーを作成して操作
        periodic_field = field.copy()

        # 各次元に対して周期境界を適用
        for axis in range(field.ndim):
            # スライス作成
            slices = [slice(None)] * field.ndim

            # 前方境界の処理（精度のため2点使用）
            for i in range(2):
                slices[axis] = i
                opposite = slices.copy()
                opposite[axis] = -2 + i
                periodic_field[tuple(slices)] = field[tuple(opposite)]

            # 後方境界の処理
            for i in range(2):
                slices[axis] = -(i + 1)
                opposite = slices.copy()
                opposite[axis] = 1 - i
                periodic_field[tuple(slices)] = field[tuple(opposite)]

        return periodic_field


class NeumannBC(BoundaryCondition):
    def get_stencil_operator(self) -> StencilOperator:
        """4次精度片側差分の係数を返す"""
        return StencilOperator(
            points=[0, 1, 2, 3, 4], coefficients=[-25 / 12, 4, -3, 4 / 3, -1 / 4]
        )

    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        """多次元ノイマン境界条件の実装"""
        neumann_field = field.copy()

        for axis in range(field.ndim):
            slices = [slice(None)] * field.ndim

            # 前方境界の処理
            for i in range(2):
                slices[axis] = i
                next_slice = slices.copy()
                next_slice[axis] = i + 1
                neumann_field[tuple(slices)] = neumann_field[tuple(next_slice)]

            # 後方境界の処理
            for i in range(2):
                slices[axis] = -(i + 1)
                prev_slice = slices.copy()
                prev_slice[axis] = -(i + 2)
                neumann_field[tuple(slices)] = neumann_field[tuple(prev_slice)]

        return neumann_field


class DirectionalBC:
    """方向ごとに異なる境界条件を適用"""

    def __init__(
        self, x_bc: BoundaryCondition, y_bc: BoundaryCondition, z_bc: BoundaryCondition
    ):
        self.conditions = [x_bc, y_bc, z_bc]

    def get_condition(self, axis: int) -> BoundaryCondition:
        """指定された方向の境界条件を取得"""
        if not 0 <= axis < len(self.conditions):
            raise ValueError(f"無効な軸: {axis}")
        return self.conditions[axis]

    def apply_all(self, field: np.ndarray) -> np.ndarray:
        """すべての方向の境界条件を適用"""
        result = field.copy()
        for axis, bc in enumerate(self.conditions):
            result = bc.apply_to_field(result)
        return result
