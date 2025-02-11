"""WENOスキームを提供するパッケージ

このパッケージは、Weighted Essentially Non-Oscillatory (WENO) スキームの
実装を提供します。WENOスキームは、不連続性を含む問題に対して高次精度の
数値解を得るための手法です。

主な機能:
- 3次、5次、7次精度のWENOスキーム
- 滑らかさ指標の計算
- 非線形重み係数の計算
- 様々なマッピング関数のサポート

Example:
    ```python
    import numpy as np
    from numerics.weno import WENO5

    # データの準備
    x = np.linspace(-1, 1, 100)
    data = np.tanh(20 * x)  # 急峻な勾配を持つデータ

    # WENO5スキームの初期化
    weno = WENO5(epsilon=1e-6, mapping="henrick")

    # データの再構成
    reconstructed = weno.reconstruct(data)
    ```
"""

from .base import WENOBase
from .coefficients import WENOCoefficients
from .smoothness import SmoothnessIndicator
from .weights import WeightCalculator
from .schemes import WENO3, WENO5, WENO7

__version__ = "1.0.0"

__all__ = [
    # スキーム
    "WENO3",
    "WENO5",
    "WENO7",
    # 基底クラスとコンポーネント
    "WENOBase",
    "WENOCoefficients",
    "SmoothnessIndicator",
    "WeightCalculator",
]