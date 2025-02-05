from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class DataIO(ABC):
    """データ入出力の基底クラス"""
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_field(self, field: ScalarField | VectorField, timestep: int) -> Path:
        """フィールドデータの保存"""
        pass

    @abstractmethod
    def load_field(self, path: Path) -> ScalarField | VectorField:
        """フィールドデータの読み込み"""
        pass

    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any], timestep: int) -> Path:
        """メタデータの保存"""
        pass

    @abstractmethod
    def load_metadata(self, path: Path) -> Dict[str, Any]:
        """メタデータの読み込み"""
        pass

    def get_timestep_path(self, timestep: int, prefix: str = "", suffix: str = "") -> Path:
        """タイムステップ用のファイルパスを生成"""
        return self.base_dir / f"{prefix}{timestep:06d}{suffix}"

    def list_timesteps(self, prefix: str = "", suffix: str = "") -> list[int]:
        """利用可能なタイムステップのリストを取得"""
        files = sorted(self.base_dir.glob(f"{prefix}*{suffix}"))
        timesteps = []
        for f in files:
            try:
                timestep = int(f.stem.replace(prefix, ""))
                timesteps.append(timestep)
            except ValueError:
                continue
        return sorted(timesteps)