"""カメラ設定を管理するモジュール

このモジュールは、3D可視化におけるカメラパラメータの
管理と変換を担当します。
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Optional

from ..core.base import ViewConfig


@dataclass
class CameraState:
    """カメラの状態を表すクラス

    位置、方向、上方向ベクトルなどを保持します。
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))

    @property
    def direction(self) -> np.ndarray:
        """視線方向を取得"""
        d = self.target - self.position
        return d / np.linalg.norm(d)

    @property
    def right(self) -> np.ndarray:
        """右方向を取得"""
        r = np.cross(self.direction, self.up)
        return r / np.linalg.norm(r)


class CameraController:
    """カメラ制御クラス

    ビュー設定とカメラ状態の変換や、
    カメラの移動・回転などの操作を提供します。
    """

    @staticmethod
    def view_to_camera(
        view: ViewConfig, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> CameraState:
        """ビュー設定からカメラ状態を計算

        Args:
            view: ビュー設定
            bounds: データの境界ボックス (最小点, 最大点)

        Returns:
            カメラ状態
        """
        # 注視点の設定
        if bounds is not None:
            min_point, max_point = bounds
            center = (min_point + max_point) / 2
            target = center
        else:
            target = np.array(view.focal_point)

        # 球面座標からカメラ位置を計算
        theta = np.radians(view.azimuth)  # 方位角
        phi = np.radians(90 - view.elevation)  # 極角（天頂角）

        # 単位球面上の位置
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # スケーリングと平行移動
        position = target + view.distance * np.array([x, y, z])

        return CameraState(position=position, target=target)

    @staticmethod
    def camera_to_view(camera: CameraState) -> ViewConfig:
        """カメラ状態からビュー設定を計算

        Args:
            camera: カメラ状態

        Returns:
            ビュー設定
        """
        # カメラから注視点へのベクトル
        direction = camera.target - camera.position
        distance = np.linalg.norm(direction)

        # 方位角と仰角を計算
        x, y, z = direction / distance

        azimuth = np.degrees(np.arctan2(y, x))
        elevation = 90 - np.degrees(np.arccos(z))

        return ViewConfig(
            elevation=elevation,
            azimuth=azimuth,
            distance=distance,
            focal_point=tuple(camera.target),
        )

    @staticmethod
    def orbit_camera(
        camera: CameraState, delta_azimuth: float, delta_elevation: float
    ) -> CameraState:
        """カメラを軌道運動させる

        Args:
            camera: 現在のカメラ状態
            delta_azimuth: 方位角の変化量（度）
            delta_elevation: 仰角の変化量（度）

        Returns:
            新しいカメラ状態
        """
        # 現在のビュー設定を取得
        view = CameraController.camera_to_view(camera)

        # 角度を更新
        view.azimuth += delta_azimuth
        view.elevation = np.clip(view.elevation + delta_elevation, -89, 89)

        # 新しいカメラ状態を計算
        return CameraController.view_to_camera(view)

    @staticmethod
    def zoom_camera(camera: CameraState, zoom_factor: float) -> CameraState:
        """カメラをズームする

        Args:
            camera: 現在のカメラ状態
            zoom_factor: ズーム倍率（1より大きい場合はズームイン）

        Returns:
            新しいカメラ状態
        """
        # 現在の距離を取得
        view = CameraController.camera_to_view(camera)

        # 距離を更新
        view.distance /= zoom_factor

        # 新しいカメラ状態を計算
        return CameraController.view_to_camera(view)

    @staticmethod
    def pan_camera(camera: CameraState, delta_x: float, delta_y: float) -> CameraState:
        """カメラをパンする

        Args:
            camera: 現在のカメラ状態
            delta_x: 水平方向の移動量
            delta_y: 垂直方向の移動量

        Returns:
            新しいカメラ状態
        """
        # 移動ベクトルを計算
        right = camera.right
        up = camera.up
        offset = delta_x * right + delta_y * up

        # カメラと注視点を移動
        new_position = camera.position + offset
        new_target = camera.target + offset

        return CameraState(position=new_position, target=new_target, up=camera.up)
