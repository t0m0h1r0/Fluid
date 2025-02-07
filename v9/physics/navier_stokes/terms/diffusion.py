"""Navier-Stokes方程式の粘性項を提供するモジュール

このモジュールは、Navier-Stokes方程式の粘性項の計算を実装します。
粘性係数が空間的に変化する場合（例：二相流体）にも対応します。
"""

import numpy as np
from typing import List, Dict, Any
from core.field import VectorField, ScalarField
from ..base_term import NavierStokesTerm

class DiffusionTerm(NavierStokesTerm):
    """粘性項クラス
    
    粘性項 ∇・(μ(∇u + ∇u^T)) を計算します。
    非一様な粘性係数に対応し、保存形式で離散化を行います。
    """
    
    def __init__(self, use_conservative: bool = True):
        """粘性項を初期化
        
        Args:
            use_conservative: 保存形式で離散化するかどうか
        """
        super().__init__(name="Diffusion")
        self.use_conservative = use_conservative
    
    def compute(self, velocity: VectorField, 
                viscosity: ScalarField = None,
                density: ScalarField = None,
                **kwargs) -> List[np.ndarray]:
        """粘性項の寄与を計算
        
        Args:
            velocity: 現在の速度場
            viscosity: 粘性係数場（Noneの場合は一様粘性を仮定）
            density: 密度場（Noneの場合は一様密度を仮定）
            **kwargs: 未使用
            
        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]
        
        dx = velocity.dx
        result = []
        
        # 粘性係数と密度の設定
        if viscosity is None:
            mu = np.ones_like(velocity.components[0].data)
        else:
            mu = viscosity.data
            
        if density is None:
            rho = np.ones_like(velocity.components[0].data)
        else:
            rho = density.data
        
        for i, v_i in enumerate(velocity.components):
            if self.use_conservative:
                # 保存形式での離散化
                # ∇・(μ(∇u + ∇u^T)) = ∂/∂xj(μ(∂ui/∂xj + ∂uj/∂xi))
                diffusion = np.zeros_like(v_i.data)
                
                for j in range(velocity.ndim):
                    # ∂ui/∂xj の計算
                    dui_dxj = np.gradient(v_i.data, dx, axis=j)
                    
                    # ∂uj/∂xi の計算
                    duj_dxi = np.gradient(velocity.components[j].data, dx, axis=i)
                    
                    # μ(∂ui/∂xj + ∂uj/∂xi) の計算
                    flux = mu * (dui_dxj + duj_dxi)
                    
                    # ∂/∂xj の計算
                    diffusion += np.gradient(flux, dx, axis=j)
                
            else:
                # 非保存形式での離散化
                # μ∇²u + ∇μ・(∇u + ∇u^T)
                
                # μ∇²u の計算
                laplacian = sum(np.gradient(np.gradient(v_i.data, dx, axis=j), 
                                          dx, axis=j) 
                              for j in range(velocity.ndim))
                diffusion = mu * laplacian
                
                # ∇μ・(∇u + ∇u^T) の計算
                grad_mu = np.array([np.gradient(mu, dx, axis=j) 
                                  for j in range(velocity.ndim)])
                
                for j in range(velocity.ndim):
                    dui_dxj = np.gradient(v_i.data, dx, axis=j)
                    duj_dxi = np.gradient(velocity.components[j].data, dx, axis=i)
                    diffusion += grad_mu[j] * (dui_dxj + duj_dxi)
            
            # 密度で割って加速度に変換
            result.append(diffusion / rho)
        
        return result
    
    def compute_timestep(self, velocity: VectorField,
                        viscosity: ScalarField = None,
                        density: ScalarField = None,
                        **kwargs) -> float:
        """粘性項による時間刻み幅の制限を計算
        
        Args:
            velocity: 現在の速度場
            viscosity: 粘性係数場
            density: 密度場
            **kwargs: 未使用
            
        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float('inf')
        
        dx = velocity.dx
        
        # 粘性係数と密度の最大・最小値を取得
        if viscosity is None:
            max_viscosity = 1.0
        else:
            max_viscosity = np.max(viscosity.data)
            
        if density is None:
            min_density = 1.0
        else:
            min_density = np.min(density.data)
        
        # 粘性項の安定条件: dt ≤ dx²/(2ν) (νは動粘性係数)
        return 0.5 * dx**2 * min_density / max_viscosity
    
    def get_diagnostics(self, velocity: VectorField,
                       viscosity: ScalarField = None,
                       density: ScalarField = None,
                       **kwargs) -> Dict[str, Any]:
        """粘性項の診断情報を取得"""
        diag = super().get_diagnostics(velocity, 
                                     viscosity=viscosity,
                                     density=density,
                                     **kwargs)
        
        # 粘性応力の計算
        stress_tensor = self._compute_stress_tensor(velocity, viscosity)
        
        # 粘性散逸の計算
        dissipation = self._compute_dissipation(velocity, stress_tensor)
        
        diag.update({
            'formulation': 'conservative' if self.use_conservative else 'non-conservative',
            'max_viscosity': np.max(viscosity.data) if viscosity is not None else 1.0,
            'viscous_dissipation': dissipation
        })
        return diag
    
    def _compute_stress_tensor(self, velocity: VectorField,
                             viscosity: ScalarField = None) -> np.ndarray:
        """粘性応力テンソルを計算"""
        dx = velocity.dx
        ndim = velocity.ndim
        
        if viscosity is None:
            mu = np.ones_like(velocity.components[0].data)
        else:
            mu = viscosity.data
        
        # 応力テンソル τij = μ(∂ui/∂xj + ∂uj/∂xi)
        stress = np.zeros((ndim, ndim) + velocity.components[0].data.shape)
        
        for i in range(ndim):
            for j in range(ndim):
                dui_dxj = np.gradient(velocity.components[i].data, dx, axis=j)
                duj_dxi = np.gradient(velocity.components[j].data, dx, axis=i)
                stress[i,j] = mu * (dui_dxj + duj_dxi)
        
        return stress
    
    def _compute_dissipation(self, velocity: VectorField,
                           stress_tensor: np.ndarray) -> float:
        """粘性散逸を計算"""
        dx = velocity.dx
        ndim = velocity.ndim
        
        # 粘性散逸 Φ = τij * (∂ui/∂xj)
        dissipation = 0.0
        for i in range(ndim):
            for j in range(ndim):
                dui_dxj = np.gradient(velocity.components[i].data, dx, axis=j)
                dissipation += np.sum(stress_tensor[i,j] * dui_dxj) * dx**ndim
        
        return 0.5 * dissipation