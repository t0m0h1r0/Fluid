import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .base import DataIO
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class CSVIO(DataIO):
    """CSV形式でのデータ入出力"""
    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self.metadata_dir = base_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_field(self, field: ScalarField | VectorField, timestep: int) -> Path:
        if isinstance(field, ScalarField):
            return self._save_scalar_field(field, timestep)
        else:
            return self._save_vector_field(field, timestep)

    def load_field(self, path: Path) -> ScalarField | VectorField:
        # メタデータの読み込み
        meta_path = self.metadata_dir / f"{path.stem}_meta.csv"
        metadata = self.load_metadata(meta_path)
        
        # データの読み込み
        data = pd.read_csv(path).values
        
        if len(data.shape) == 3:  # スカラー場
            field = ScalarField(metadata)
            field.data = data
        else:  # ベクトル場
            field = VectorField(metadata)
            components = []
            for i in range(data.shape[-1]):
                components.append(data[..., i])
            field.data = components
            
        return field

    def save_metadata(self, metadata: Dict[str, Any], timestep: int) -> Path:
        path = self.metadata_dir / f"metadata_{timestep:06d}.csv"
        
        # メタデータをDataFrameに変換
        meta_dict = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value.flatten()):
                    meta_dict[f"{key}_{i}"] = v
            else:
                meta_dict[key] = value
                
        pd.DataFrame([meta_dict]).to_csv(path, index=False)
        return path

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        df = pd.read_csv(path)
        metadata = {}
        
        # 1行目のデータを辞書に変換
        for column in df.columns:
            if "_" in column:  # 配列データの復元
                key, idx = column.rsplit("_", 1)
                if key not in metadata:
                    metadata[key] = []
                metadata[key].append(df[column].iloc[0])
            else:
                metadata[column] = df[column].iloc[0]
                
        # 配列データをnumpy配列に変換
        for key in list(metadata.keys()):
            if isinstance(metadata[key], list):
                metadata[key] = np.array(metadata[key])
                
        return metadata

    def _save_scalar_field(self, field: ScalarField, timestep: int) -> Path:
        path = self.get_timestep_path(timestep, suffix=".csv")
        
        # データの保存
        df = pd.DataFrame(field.data)
        df.to_csv(path, index=False)
        
        # メタデータの保存
        meta_path = self.metadata_dir / f"{path.stem}_meta.csv"
        self.save_metadata(field.metadata.__dict__, timestep)
        
        return path

    def _save_vector_field(self, field: VectorField, timestep: int) -> Path:
        path = self.get_timestep_path(timestep, suffix=".csv")
        
        # 3次元データを2次元に変換
        data = np.stack(field.data, axis=-1)
        shape = data.shape
        
        # データの保存
        df = pd.DataFrame(data.reshape(-1, shape[-1]))
        df.columns = [f"component_{i}" for i in range(shape[-1])]
        df.to_csv(path, index=False)
        
        # メタデータの保存（形状情報を含む）
        metadata = field.metadata.__dict__.copy()
        metadata['original_shape'] = shape
        meta_path = self.metadata_dir / f"{path.stem}_meta.csv"
        self.save_metadata(metadata, timestep)
        
        return path

    def save_timeseries(self, 
                       data: Dict[str, list], 
                       timestep: int,
                       columns: Optional[list[str]] = None) -> Path:
        """時系列データの保存"""
        path = self.get_timestep_path(timestep, prefix="timeseries_", suffix=".csv")
        
        df = pd.DataFrame(data)
        if columns:
            df.columns = columns
            
        df.to_csv(path, index=False)
        return path

    def load_timeseries(self, path: Path) -> pd.DataFrame:
        """時系列データの読み込み"""
        return pd.read_csv(path)