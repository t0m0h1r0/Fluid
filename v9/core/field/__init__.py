# core/__init__.py
from .field import Field
from .scalar import ScalarField
from .vector import VectorField
from .conserved import ConservedField

__all__ = ["Field", "ScalarField", "VectorField", "ConservedField"]

# プロジェクトのディレクトリ構造:
"""
project_root/
├── core/
│   ├── __init__.py
│   ├── field.py      # 基底Field クラス
│   ├── scalar.py     # ScalarField クラス
│   ├── vector.py     # VectorField クラス
│   └── conserved.py  # ConservedField クラス
├── simulations/
│   ├── __init__.py
│   ├── state.py
│   └── ...
└── main.py
"""
