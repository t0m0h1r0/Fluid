import os
import numpy as np
import h5py
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from core.field import Field, VectorField

class DataManager:
    """データの入出力を管理するクラス"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.plot_dir = self.output_dir / "plots"
        self.log_dir = self.output_dir / "logs"
        
        # ディレクトリの作成
        for directory in [self.output_dir, self.data_dir, 
                         self.plot_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # ログファイルの初期化
        self.log_file = self.log_dir / f"simulation_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # 実行時のメタデータを記録
        self._save_run_metadata()
    
    def save_state(self, step: int, time: float, fields: Dict[str, Any],
                  **kwargs) -> None:
        """状態の保存
        
        Args:
            step: タイムステップ
            time: 時刻
            fields: 保存する場のデータ
            **kwargs: 追加のメタデータ
        """
        # ファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"state_{step:06d}_{timestamp}.h5"
        
        with h5py.File(filename, 'w') as f:
            # メタデータ
            f.attrs['step'] = step
            f.attrs['time'] = time
            f.attrs['timestamp'] = timestamp
            
            # 追加のメタデータ
            for key, value in kwargs.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
            
            # フィールドデータ
            for name, field in fields.items():
                if isinstance(field, Field):
                    f.create_dataset(name, data=field.data)
                elif isinstance(field, VectorField):
                    group = f.create_group(name)
                    for i, component in enumerate(field.components):
                        group.create_dataset(f"component_{i}", 
                                          data=component.data)
                elif isinstance(field, np.ndarray):
                    f.create_dataset(name, data=field)
                elif isinstance(field, dict):
                    group = f.create_group(name)
                    for key, value in field.items():
                        if isinstance(value, (int, float, str)):
                            group.attrs[key] = value
    
    def load_state(self, filename: str) -> Dict[str, Any]:
        """状態の読み込み"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # メタデータの読み込み
            metadata = dict(f.attrs)
            
            # フィールドデータの読み込み
            fields = {}
            for name in f.keys():
                if isinstance(f[name], h5py.Dataset):
                    # 単一のデータセット
                    fields[name] = f[name][...]
                else:
                    # グループ（ベクトル場など）
                    if 'component_0' in f[name]:
                        # ベクトル場の再構築
                        components = []
                        i = 0
                        while f'component_{i}' in f[name]:
                            components.append(f[name][f'component_{i}'][...])
                            i += 1
                        fields[name] = components
                    else:
                        # その他の階層的データ
                        fields[name] = dict(f[name].attrs)
            
            return {'metadata': metadata, 'fields': fields}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """設定の保存"""
        filename = self.output_dir / "config.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self) -> Dict[str, Any]:
        """設定の読み込み"""
        filename = self.output_dir / "config.json"
        if not filename.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {filename}")
            
        with open(filename, 'r') as f:
            return json.load(f)
    
    def _save_run_metadata(self) -> None:
        """実行時のメタデータを保存"""
        metadata = {
            'start_time': datetime.now().isoformat(),
            'output_dir': str(self.output_dir.absolute()),
            'python_version': os.sys.version,
            'platform': os.sys.platform
        }
        
        filename = self.output_dir / "run_metadata.json"
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_log(self, message: str, level: str = 'INFO') -> None:
        """ログの保存"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
    
    def clean_old_files(self, max_files: int = 1000) -> None:
        """古いファイルの削除"""
        def clean_directory(directory: Path, pattern: str):
            files = sorted(directory.glob(pattern))
            if len(files) > max_files:
                for file in files[:-max_files]:
                    file.unlink()
                self.save_log(
                    f"{len(files) - max_files}個の古いファイルを削除しました: {directory}"
                )
        
        # 各ディレクトリの古いファイルを削除
        clean_directory(self.data_dir, "state_*.h5")
        clean_directory(self.plot_dir, "*.png")
        clean_directory(self.log_dir, "*.log")
    
    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """最新の状態を取得"""
        state_files = sorted(self.data_dir.glob("state_*.h5"))
        if not state_files:
            return None
            
        return self.load_state(state_files[-1].name)
    
    def get_state_files(self) -> List[Path]:
        """全ての状態ファイルを取得"""
        return sorted(self.data_dir.glob("state_*.h5"))
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報の取得"""
        state_files = self.get_state_files()
        
        stats = {
            'total_files': len(state_files),
            'disk_usage': sum(f.stat().st_size for f in state_files),
            'first_step': None,
            'last_step': None,
            'simulation_time': None
        }
        
        if state_files:
            first_state = self.load_state(state_files[0].name)
            last_state = self.load_state(state_files[-1].name)
            
            stats.update({
                'first_step': first_state['metadata']['step'],
                'last_step': last_state['metadata']['step'],
                'simulation_time': last_state['metadata']['time']
            })
        
        return stats
    
    def export_to_vtk(self, fields: Dict[str, Any], filename: str) -> None:
        """VTKフォーマットでエクスポート"""
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            self.save_log("VTKライブラリがインストールされていません", level='ERROR')
            return
        
        # VTKグリッドの作成
        grid = vtk.vtkStructuredGrid()
        
        # 座標の設定
        if isinstance(next(iter(fields.values())), (Field, VectorField)):
            shape = next(iter(fields.values())).shape
        else:
            shape = fields[next(iter(fields))].shape
        
        points = vtk.vtkPoints()
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    points.InsertNextPoint(i, j, k)
        
        grid.SetPoints(points)
        grid.SetDimensions(*shape)
        
        # フィールドデータの追加
        for name, field in fields.items():
            if isinstance(field, Field):
                array = numpy_support.numpy_to_vtk(field.data.flatten())
                array.SetName(name)
                grid.GetPointData().AddArray(array)
            elif isinstance(field, VectorField):
                vectors = np.stack([c.data for c in field.components], axis=-1)
                array = numpy_support.numpy_to_vtk(vectors.reshape(-1, 3))
                array.SetName(name)
                grid.GetPointData().AddArray(array)
        
        # ファイルの保存
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(self.output_dir / f"{filename}.vts"))
        writer.SetInputData(grid)
        writer.Write()
