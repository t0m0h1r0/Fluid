import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from .io_base import DataIO
from core.field.base import FieldMetadata
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class HDF5IO(DataIO):
    """HDF5形式でのデータ入出力"""
    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        
    def save_field(self, field: ScalarField | VectorField, timestep: int) -> Path:
        path = self.get_timestep_path(timestep, suffix=".h5")
        
        with h5py.File(path, 'w') as f:
            # メタデータの保存
            meta = f.create_group('metadata')
            self._save_metadata_to_group(meta, field.metadata)
            
            # データの保存
            if isinstance(field, ScalarField):
                f.create_dataset('data', data=field.data)
                f.attrs['field_type'] = 'scalar'
            else:
                data = f.create_group('data')
                for i, component in enumerate(field.data):
                    data.create_dataset(f'component_{i}', data=component)
                f.attrs['field_type'] = 'vector'
            
            # タイムスタンプの保存
            f.attrs['saved_at'] = datetime.now().isoformat()
            
        return path

    def load_field(self, path: Path) -> ScalarField | VectorField:
        with h5py.File(path, 'r') as f:
            # メタデータの読み込み
            metadata = self._load_metadata_from_group(f['metadata'])
            
            # データの読み込み
            if f.attrs['field_type'] == 'scalar':
                field = ScalarField(metadata)
                field.data = f['data'][:]
            else:
                field = VectorField(metadata)
                components = []
                data = f['data']
                for i in range(len(data)):
                    components.append(data[f'component_{i}'][:])
                field.data = components
            
            return field

    def save_metadata(self, metadata: Dict[str, Any], timestep: int) -> Path:
        path = self.get_timestep_path(timestep, prefix="meta_", suffix=".h5")
        
        with h5py.File(path, 'w') as f:
            self._save_metadata_to_group(f, metadata)
            f.attrs['saved_at'] = datetime.now().isoformat()
            
        return path

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        with h5py.File(path, 'r') as f:
            return self._load_metadata_from_group(f)

    def _save_metadata_to_group(self, group: h5py.Group, metadata: Dict[str, Any]) -> None:
        """メタデータをHDF5グループに保存"""
        for key, value in metadata.items():
            if isinstance(value, (np.ndarray, list)):
                group.create_dataset(key, data=value)
            else:
                group.attrs[key] = value

    def _load_metadata_from_group(self, group: h5py.Group) -> Dict[str, Any]:
        """HDF5グループからメタデータを読み込み"""
        metadata = {}
        
        # データセットの読み込み
        for key in group.keys():
            metadata[key] = group[key][:]
        
        # 属性の読み込み
        for key in group.attrs.keys():
            metadata[key] = group.attrs[key]
            
        return metadata