from abc import ABC, abstractmethod
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from core.field import Field, VectorField

class TimeIntegrator(ABC):
    """時間積分の基底クラス"""
    
    @abstractmethod
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        """1ステップの時間発展
        
        Args:
            field: 更新する場
            rhs_func: 右辺を計算する関数
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ
            
        Returns:
            更新された場
        """
        pass

class RungeKutta4(TimeIntegrator):
    """4次のルンゲ・クッタ法"""
    
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        # RK4の係数
        a = [0.0, 0.5, 0.5, 1.0]
        b = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
        
        k = []
        temp_field = field.copy()
        
        for i in range(4):
            # 右辺の評価
            if i == 0:
                k.append(rhs_func(field, **kwargs))
            else:
                temp_field.data = field.data + a[i] * dt * k[i-1]
                k.append(rhs_func(temp_field, **kwargs))
        
        # 解の更新
        new_field = field.copy()
        new_field.data = field.data + dt * sum(b[i] * k[i] for i in range(4))
        
        return new_field

class AdamsBashforth(TimeIntegrator):
    """Adams-Bashforth法"""
    
    def __init__(self, order: int = 3):
        """
        Args:
            order: スキームの次数（2-4）
        """
        if order not in [2, 3, 4]:
            raise ValueError("次数は2, 3, 4のいずれかである必要があります")
        
        self.order = order
        self.previous_rhs = []
        
        # 係数の設定
        if order == 2:
            self.coeffs = [3/2, -1/2]
        elif order == 3:
            self.coeffs = [23/12, -16/12, 5/12]
        else:  # order == 4
            self.coeffs = [55/24, -59/24, 37/24, -9/24]
    
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        # 右辺の計算
        current_rhs = rhs_func(field, **kwargs)
        
        # 初期ステップの処理
        if len(self.previous_rhs) < self.order - 1:
            # Euler法を使用
            new_field = field.copy()
            new_field.data = field.data + dt * current_rhs
            self.previous_rhs.append(current_rhs)
            return new_field
        
        # Adams-Bashforth法による更新
        new_field = field.copy()
        rhs_sum = sum(c * r for c, r in zip(self.coeffs[:-1], 
                                          [current_rhs] + self.previous_rhs))
        new_field.data = field.data + dt * rhs_sum
        
        # 履歴の更新
        self.previous_rhs = [current_rhs] + self.previous_rhs[:-1]
        
        return new_field

class AdamsMoulton(TimeIntegrator):
    """Adams-Moulton法（予測子-修正子法）"""
    
    def __init__(self, order: int = 4, max_iterations: int = 3, 
                 tolerance: float = 1e-6):
        """
        Args:
            order: スキームの次数（2-4）
            max_iterations: 最大反復回数
            tolerance: 収束判定の閾値
        """
        if order not in [2, 3, 4]:
            raise ValueError("次数は2, 3, 4のいずれかである必要があります")
            
        self.order = order
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.previous_rhs = []
        
        # 予測子にはAdams-Bashforthを使用
        self.predictor = AdamsBashforth(order=order)
        
        # 係数の設定
        if order == 2:
            self.coeffs = [5/12, 8/12, -1/12]
        elif order == 3:
            self.coeffs = [9/24, 19/24, -5/24, 1/24]
        else:  # order == 4
            self.coeffs = [251/720, 646/720, -264/720, 106/720, -19/720]
    
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        # 予測子ステップ
        predicted_field = self.predictor.step(field, rhs_func, dt, **kwargs)
        
        # 初期ステップの処理
        if len(self.previous_rhs) < self.order - 1:
            self.previous_rhs.append(rhs_func(field, **kwargs))
            return predicted_field
        
        # 修正子ステップ
        current_field = predicted_field
        for _ in range(self.max_iterations):
            old_field = current_field.copy()
            
            # 右辺の評価
            current_rhs = rhs_func(current_field, **kwargs)
            
            # Adams-Moulton法による更新
            rhs_sum = sum(c * r for c, r in zip(self.coeffs[:-1],
                         [current_rhs] + self.previous_rhs))
            current_field.data = field.data + dt * rhs_sum
            
            # 収束判定
            if np.max(np.abs(current_field.data - old_field.data)) < self.tolerance:
                break
        
        # 履歴の更新
        self.previous_rhs = [current_rhs] + self.previous_rhs[:-1]
        
        return current_field

class CrankNicolson(TimeIntegrator):
    """Crank-Nicolson法"""
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        current_field = field.copy()
        
        for _ in range(self.max_iterations):
            old_field = current_field.copy()
            
            # 中点での右辺の評価
            midpoint = field.copy()
            midpoint.data = 0.5 * (field.data + current_field.data)
            rhs_mid = rhs_func(midpoint, **kwargs)
            
            # Crank-Nicolson更新
            current_field.data = field.data + dt * rhs_mid
            
            # 収束判定
            if np.max(np.abs(current_field.data - old_field.data)) < self.tolerance:
                break
        
        return current_field

class AdaptiveTimeIntegrator(TimeIntegrator):
    """適応的時間積分"""
    
    def __init__(self, base_integrator: TimeIntegrator, safety_factor: float = 0.9,
                 min_dt: float = 1e-6, max_dt: float = 1.0):
        """
        Args:
            base_integrator: 基本となる時間積分法
            safety_factor: 安全係数
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
        """
        self.integrator = base_integrator
        self.safety_factor = safety_factor
        self.min_dt = min_dt
        self.max_dt = max_dt
    
    def step(self, field: Field, rhs_func: Callable, dt: float, **kwargs) -> Field:
        while True:
            try:
                # 通常のステップ
                new_field = self.integrator.step(field, rhs_func, dt, **kwargs)
                
                # CFLチェック
                if 'velocity' in kwargs:
                    velocity = kwargs['velocity']
                    dx = kwargs.get('dx', 1.0)
                    cfl = np.max(np.abs(velocity)) * dt / dx
                    if cfl > 1.0:
                        dt *= self.safety_factor
                        continue
                
                return new_field
                
            except (RuntimeError, ValueError) as e:
                # エラーが発生した場合は時間刻み幅を減少
                dt *= 0.5
                if dt < self.min_dt:
                    raise RuntimeError("時間刻み幅が最小値を下回りました") from e
                continue
        
    def compute_next_dt(self, field: Field, new_field: Field, dt: float,
                       tolerance: float = 1e-3) -> float:
        """次の時間刻み幅を計算"""
        error = np.max(np.abs(new_field.data - field.data))
        
        if error > 0:
            dt_new = self.safety_factor * dt * (tolerance / error) ** 0.5
            return np.clip(dt_new, self.min_dt, self.max_dt)
        else:
            return self.max_dt
