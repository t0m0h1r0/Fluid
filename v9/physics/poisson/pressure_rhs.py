"""圧力ポアソン方程式の右辺を計算するモジュール

このモジュールは、レベルセット法を用いた二層流のNavier-Stokes方程式から
導出される圧力ポアソン方程式の右辺を計算します。
"""

from typing import List, Protocol, Dict, Any
import numpy as np
from physics.levelset import LevelSetField
from core.field import VectorField, ScalarField
from physics.properties import PropertiesManager


class SourceTermComputer(Protocol):
    """ポアソン方程式の源泉項を計算するインターフェース"""
    
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> np.ndarray:
        """源泉項を計算
        
        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            
        Returns:
            計算された源泉項
        """
        ...


class AdvectionTermComputer(SourceTermComputer):
    """移流項の計算を行うクラス"""
    
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> np.ndarray:
        """移流項 ∇⋅[−ρ(u⋅∇)u] を計算"""
        # 密度場の取得
        density = properties.get_density(levelset)
        
        # (u⋅∇)u の計算
        convection = np.zeros_like(velocity.components[0].data)
        for i, u_i in enumerate(velocity.components):
            for j, u_j in enumerate(velocity.components):
                convection += u_j.data * u_i.gradient(j)
                
        # ρ(u⋅∇)u の発散を計算
        result = np.zeros_like(convection)
        for i in range(velocity.ndim):
            result += np.gradient(-density * convection, velocity.dx, axis=i)
            
        return result


class ViscousTermComputer(SourceTermComputer):
    """粘性項の計算を行うクラス"""
    
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> np.ndarray:
        """粘性項 ∇⋅[∇⋅(2μD)] を計算"""
        # 粘性係数場の取得
        viscosity = properties.get_viscosity(levelset)
        dx = velocity.dx
        
        # 変形速度テンソルの各成分を計算
        D = np.zeros((velocity.ndim, velocity.ndim) + velocity.shape)
        for i in range(velocity.ndim):
            for j in range(velocity.ndim):
                D[i,j] = 0.5 * (
                    np.gradient(velocity.components[i].data, dx, axis=j) +
                    np.gradient(velocity.components[j].data, dx, axis=i)
                )
        
        # ∇⋅(2μD) の計算
        stress = np.zeros_like(D)
        for i in range(velocity.ndim):
            for j in range(velocity.ndim):
                stress[i,j] = 2 * viscosity * D[i,j]
        
        # 応力テンソルの発散を計算
        result = np.zeros_like(velocity.components[0].data)
        for i in range(velocity.ndim):
            for j in range(velocity.ndim):
                result += np.gradient(stress[i,j], dx, axis=j)
                
        return result


class GravityTermComputer(SourceTermComputer):
    """重力項の計算を行うクラス"""
    
    def __init__(self, gravity: float = 9.81, direction: int = -1):
        """
        Args:
            gravity: 重力加速度
            direction: 重力方向（-1: 負のz方向）
        """
        self.gravity = gravity
        self.direction = direction
    
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> np.ndarray:
        """重力項 ∇⋅(ρg) を計算"""
        # 密度勾配の計算
        density = properties.get_density(levelset)
        density_grad = np.gradient(density, velocity.dx, axis=abs(self.direction))
        
        # 重力方向の考慮
        if self.direction < 0:
            density_grad = -density_grad
            
        return density_grad * self.gravity


class SurfaceTensionTermComputer(SourceTermComputer):
    """表面張力項の計算を行うクラス"""
    
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> np.ndarray:
        """表面張力項 ∇⋅[σκδ(ϕ)n] を計算"""
        # 表面張力係数の取得
        sigma = properties.get_surface_tension_coefficient()
        if sigma is None or sigma == 0:
            return np.zeros_like(velocity.components[0].data)
            
        # 界面の法線と曲率を計算
        kappa = levelset.curvature()
        delta = levelset.delta()
        
        # 表面力の計算
        force = sigma * kappa * delta
        
        # 発散の計算
        result = np.zeros_like(force)
        for i in range(velocity.ndim):
            grad_phi = levelset.gradient(i)
            grad_norm = np.sqrt(sum(levelset.gradient(j)**2 
                              for j in range(velocity.ndim)))
            normal = grad_phi / (grad_norm + 1e-10)  # ゼロ除算防止
            result += np.gradient(force * normal, velocity.dx, axis=i)
            
        return result


class PoissonRHSComputer:
    """圧力ポアソン方程式の右辺を計算するクラス"""
    
    def __init__(self):
        """計算コンポーネントの初期化"""
        self.source_terms: List[SourceTermComputer] = [
            AdvectionTermComputer(),
            ViscousTermComputer(),
            GravityTermComputer(),
            SurfaceTensionTermComputer()
        ]
        
    def compute(self, velocity: VectorField, levelset: LevelSetField, 
                properties: PropertiesManager) -> ScalarField:
        """圧力ポアソン方程式の右辺を計算
        
        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            
        Returns:
            計算された右辺のスカラー場
        """
        result = ScalarField(velocity.shape, velocity.dx)
        
        # 各項の寄与を合計
        for term in self.source_terms:
            result.data += term.compute(velocity, levelset, properties)
            
        return result
    
    def get_diagnostics(self, velocity: VectorField, levelset: LevelSetField,
                       properties: PropertiesManager) -> Dict[str, Any]:
        """診断情報を取得
        
        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            
        Returns:
            各項の寄与などの診断情報
        """
        diagnostics = {}
        
        # 各項の最大値と平均値を記録
        for i, term in enumerate(self.source_terms):
            contribution = term.compute(velocity, levelset, properties)
            diagnostics[f"term_{i}_max"] = float(np.max(np.abs(contribution)))
            diagnostics[f"term_{i}_mean"] = float(np.mean(np.abs(contribution)))
            
        return diagnostics