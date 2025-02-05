from dataclasses import dataclass
from typing import Tuple, List, Union

@dataclass
class FieldMetadata:
    """場の付加情報"""
    name: str
    unit: str
    domain_size: Union[Tuple[float, float, float], List[float]]
    resolution: Union[Tuple[int, int, int], List[int]]
    time: float = 0.0

    def __post_init__(self):
        # タプルへの変換を確実に行う
        if isinstance(self.domain_size, list):
            self.domain_size = tuple(self.domain_size)
        if isinstance(self.resolution, list):
            self.resolution = tuple(self.resolution)

    @property
    def dx(self) -> Tuple[float, float, float]:
        """格子間隔"""
        return tuple(d/n for d, n in zip(self.domain_size, self.resolution))