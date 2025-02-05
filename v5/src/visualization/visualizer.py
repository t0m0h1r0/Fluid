# visualization/visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Tuple
import os
import json


class SimulationVisualizer:
    def __init__(self, output_dir: str = "output"):
        """可視化クラスの初期化

        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        self.animation_dir = os.path.join(output_dir, "animations")

        # ディレクトリの作成
        for directory in [self.snapshot_dir, self.animation_dir]:
            os.makedirs(directory, exist_ok=True)

        # 描画スタイルの設定
        plt.style.use("default")  # seabornの代わりにデフォルトスタイルを使用

        # フォントの設定
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

        # 図のサイズとDPIの設定
        self.default_figsize = (15, 10)
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300

        # アニメーション用のバッファ
        self.animation_buffer = []
        self.max_buffer_size = 100

        # カラーマップの設定
        self.colormaps = {
            "phase": "RdBu",
            "velocity": "viridis",
            "pressure": "coolwarm",
            "vorticity": "RdYlBu",
        }

    def create_visualization(
        self, data: Dict[str, np.ndarray], timestep: int, save: bool = True
    ) -> None:
        """複合可視化の作成"""
        # マルチプロット図の作成
        fig = plt.figure(figsize=self.default_figsize)

        # 2D断面プロット
        self._plot_2d_slices(fig, data, 231)  # 位置231は2x3グリッドの1番目

        # 3D等値面プロット
        self._plot_3d_isosurface(fig, data, 232)

        # 速度場プロット
        self._plot_velocity_field(fig, data, 233)

        # 渦度プロット
        self._plot_vorticity(fig, data, 234)

        # 統計情報プロット
        self._plot_statistics(fig, data, 235)

        # エネルギースペクトルプロット
        self._plot_energy_spectrum(fig, data, 236)

        # 全体のタイトル設定
        fig.suptitle(
            f"シミュレーション時間: {data['stats']['total_time']:.3f}s\n"
            f"タイムステップ: {timestep}"
        )

        plt.tight_layout()

        if save:
            # 画像として保存
            filename = os.path.join(
                self.snapshot_dir, f"visualization_{timestep:06d}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

        # アニメーション用にバッファに追加
        if len(self.animation_buffer) >= self.max_buffer_size:
            self.animation_buffer.pop(0)
        self.animation_buffer.append((fig, data))

    def _plot_2d_slices(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """2D断面プロットの作成"""
        ax = fig.add_subplot(position)

        # 相場の断面
        phi_slice = self._get_middle_slice(data["phi"], axis=2)
        im = ax.imshow(
            phi_slice.T, cmap=self.colormaps["phase"], origin="lower", aspect="equal"
        )
        plt.colorbar(im, ax=ax, label="相場")

        # 速度ベクトルの重ね描き
        if "velocity" in data:
            u_slice = self._get_middle_slice(data["velocity"][0], axis=2)
            v_slice = self._get_middle_slice(data["velocity"][1], axis=2)

            # ベクトル場を間引いて表示
            stride = 4
            x, y = np.meshgrid(
                np.arange(0, u_slice.shape[0], stride),
                np.arange(0, u_slice.shape[1], stride),
            )
            ax.quiver(
                x,
                y,
                u_slice[::stride, ::stride].T,
                v_slice[::stride, ::stride].T,
                scale=50,
                alpha=0.5,
            )

        ax.set_title("相場と速度ベクトル")

    def _plot_3d_isosurface(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """3D等値面プロットの作成"""
        from skimage import measure

        ax = fig.add_subplot(position, projection="3d")

        # 相場データの取得
        phi = data["phi"]

        try:
            # 等値面の計算（レベル0で界面を検出）
            verts, faces, _, _ = measure.marching_cubes(
                phi,
                level=0.0,
                allow_degenerate=False,
                spacing=(1.0, 1.0, 1.0),  # グリッド間隔を指定
            )

            # 等値面の描画
            if len(verts) > 0:
                ax.plot_trisurf(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    triangles=faces,
                    cmap=self.colormaps["phase"],
                    alpha=0.8,
                )

        except ValueError as e:
            print(f"警告: 等値面の計算に失敗しました: {e}")
            print(f"相場の範囲: [{phi.min():.3f}, {phi.max():.3f}]")

        ax.set_title("界面等値面")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ビューの設定
        ax.view_init(elev=30, azim=45)

    def _plot_velocity_field(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """速度場プロットの作成"""
        ax = fig.add_subplot(position)

        # 速度の大きさを計算
        if "velocity" in data:
            velocity_magnitude = np.sqrt(sum(v**2 for v in data["velocity"]))

            # 中央断面での速度場
            vel_slice = self._get_middle_slice(velocity_magnitude, axis=1)

            im = ax.imshow(
                vel_slice.T,
                cmap=self.colormaps["velocity"],
                origin="lower",
                aspect="equal",
            )
            plt.colorbar(im, ax=ax, label="速度の大きさ")
            ax.set_title("速度場")
        else:
            ax.text(
                0.5,
                0.5,
                "速度データなし",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("速度場（データなし）")

    def _plot_vorticity(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """渦度プロットの作成"""
        ax = fig.add_subplot(position)

        if "velocity" in data:
            # 渦度の計算
            dx = dy = dz = 1.0  # 格子間隔

            dudy, dudz = np.gradient(data["velocity"][0], dy, dz, axis=(1, 2))
            dvdx, dvdz = np.gradient(data["velocity"][1], dx, dz, axis=(0, 2))
            dwdx, dwdy = np.gradient(data["velocity"][2], dx, dy, axis=(0, 1))

            vorticity_x = dwdy - dvdz
            vorticity_y = dudz - dwdx
            vorticity_z = dvdx - dudy

            # 渦度の大きさ
            vorticity_magnitude = np.sqrt(
                vorticity_x**2 + vorticity_y**2 + vorticity_z**2
            )

            # 中央断面での渦度
            vort_slice = self._get_middle_slice(vorticity_magnitude, axis=1)

            im = ax.imshow(
                vort_slice.T,
                cmap=self.colormaps["vorticity"],
                origin="lower",
                aspect="equal",
            )
            plt.colorbar(im, ax=ax, label="渦度の大きさ")
            ax.set_title("渦度場")
        else:
            ax.text(
                0.5,
                0.5,
                "速度データなし",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("渦度場（データなし）")

    def _plot_statistics(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """統計情報プロットの作成"""
        ax = fig.add_subplot(position)

        if "stats" in data:
            stats = data["stats"]

            # 統計情報のテキスト表示
            stat_text = (
                f"タイムステップ数: {stats['timesteps']}\n"
                f"平均dt: {stats['avg_timestep']:.2e}s\n"
                f"最大速度: {stats['max_velocity']:.2f}m/s\n"
                f"エネルギー保存: {stats['energy_conservation']:.3f}\n"
                f"質量保存: {stats['mass_conservation']:.3f}\n"
                f"界面エネルギー: {stats.get('interface_energy', 0.0):.3f}"
            )

            ax.text(
                0.5,
                0.5,
                stat_text,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "統計データなし",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        ax.set_title("シミュレーション統計")
        ax.axis("off")

    def _plot_energy_spectrum(
        self, fig: plt.Figure, data: Dict[str, np.ndarray], position: int
    ) -> None:
        """エネルギースペクトルプロットの作成"""
        ax = fig.add_subplot(position)

        if "velocity" in data:
            # 運動エネルギースペクトルの計算
            k, E = self._compute_energy_spectrum(data)

            ax.loglog(k, E, "k-", label="運動エネルギー")
            ax.loglog(k, k ** (-5 / 3), "r--", label="k^(-5/3)")

            ax.set_xlabel("波数 k")
            ax.set_ylabel("E(k)")
            ax.set_title("エネルギースペクトル")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(
                0.5,
                0.5,
                "速度データなし",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("エネルギースペクトル（データなし）")

    def _compute_energy_spectrum(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """運動エネルギースペクトルの計算"""
        # 3D FFTの計算
        u_hat = np.fft.fftn(data["velocity"][0])
        v_hat = np.fft.fftn(data["velocity"][1])
        w_hat = np.fft.fftn(data["velocity"][2])

        # パワースペクトル密度
        E_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2 + np.abs(w_hat) ** 2)

        # 波数の計算
        kx = np.fft.fftfreq(E_hat.shape[0])
        ky = np.fft.fftfreq(E_hat.shape[1])
        kz = np.fft.fftfreq(E_hat.shape[2])

        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)

        # シェル平均
        k_bins = np.linspace(0, np.max(K), 50)
        E = np.zeros_like(k_bins[:-1])

        for i in range(len(k_bins) - 1):
            mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
            if mask.any():
                E[i] = np.mean(E_hat[mask])

        return k_bins[:-1], E

    def create_animation(self, filename: str) -> None:
        """アニメーションの作成と保存"""
        if not self.animation_buffer:
            return

        fig = plt.figure(figsize=self.default_figsize)

        def update(frame):
            plt.clf()
            frame_fig, frame_data = self.animation_buffer[frame]
            self.create_visualization(frame_data, frame, save=False)

        anim = FuncAnimation(
            fig, update, frames=len(self.animation_buffer), interval=100
        )

        anim.save(os.path.join(self.animation_dir, filename), writer="pillow", fps=10)
        plt.close()

    @staticmethod
    def _get_middle_slice(array: np.ndarray, axis: int) -> np.ndarray:
        """配列の中央断面を取得"""
        slicing = [slice(None)] * array.ndim
        slicing[axis] = array.shape[axis] // 2
        return array[tuple(slicing)]

    def get_animation_frames(self) -> int:
        """アニメーションのフレーム数を取得"""
        return len(self.animation_buffer)

    def clear_animation_buffer(self):
        """アニメーションバッファをクリア"""
        self.animation_buffer.clear()

    def _setup_axis_labels(
        self, ax: plt.Axes, xlabel: str = "X", ylabel: str = "Y", zlabel: str = None
    ):
        """軸ラベルの設定"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if zlabel and hasattr(ax, "set_zlabel"):
            ax.set_zlabel(zlabel)
        ax.grid(True)

    def save_metadata(self, metadata: dict, timestep: int):
        """メタデータの保存"""
        metadata_file = os.path.join(self.snapshot_dir, f"metadata_{timestep:06d}.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def add_colorbar(self, ax: plt.Axes, im, label: str):
        """カラーバーの追加"""
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)
        return cbar
