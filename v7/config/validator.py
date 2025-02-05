from typing import List
from .base import Config

class ConfigValidator:
    """設定の妥当性検証"""
    
    @staticmethod
    def validate(config: Config) -> List[str]:
        errors = []
        
        # 物理パラメータの検証
        errors.extend(ConfigValidator._validate_physical(config))
        
        # ドメイン設定の検証
        errors.extend(ConfigValidator._validate_domain(config))
        
        # 初期条件の検証
        errors.extend(ConfigValidator._validate_initial_conditions(config))
        
        # 数値計算設定の検証
        errors.extend(ConfigValidator._validate_numerical(config))
        
        # 境界条件の検証
        errors.extend(ConfigValidator._validate_boundary_conditions(config))
        
        return errors

    @staticmethod
    def _validate_physical(config: Config) -> List[str]:
        errors = []
        
        # 相のチェック
        if not config.physical.phases:
            errors.append("少なくとも1つの相を定義する必要があります")
        
        for phase in config.physical.phases:
            if phase.density <= 0:
                errors.append(f"相 {phase.name} の密度は正の値である必要があります")
            
            if phase.viscosity <= 0:
                errors.append(f"相 {phase.name} の粘性係数は正の値である必要があります")
        
        # 重力のチェック
        if config.physical.gravity < 0:
            errors.append("重力は非負である必要があります")
        
        return errors

    @staticmethod
    def _validate_domain(config: Config) -> List[str]:
        errors = []
        
        # グリッド数のチェック
        for dim, val in [
            ('Nx', config.domain.Nx),
            ('Ny', config.domain.Ny),
            ('Nz', config.domain.Nz)
        ]:
            if val <= 0:
                errors.append(f"{dim}は正の整数である必要があります")
        
        # ドメインサイズのチェック
        for dim, val in [
            ('Lx', config.domain.Lx),
            ('Ly', config.domain.Ly),
            ('Lz', config.domain.Lz)
        ]:
            if val <= 0:
                errors.append(f"{dim}は正の値である必要があります")
        
        # グリッド数の制限
        max_points = 1e7  # 1000万点
        total_points = config.domain.Nx * config.domain.Ny * config.domain.Nz
        if total_points > max_points:
            errors.append(f"総グリッド数が多すぎます: {total_points} > {max_points}")
        
        return errors

    @staticmethod
    def _validate_initial_conditions(config: Config) -> List[str]:
        errors = []
        
        # 層の検証
        if not config.initial_condition.layers:
            errors.append("少なくとも1つの層を定義する必要があります")
        
        for layer in config.initial_condition.layers:
            # 層の高さチェック
            if layer.z_range[0] >= layer.z_range[1]:
                errors.append(f"相 {layer.phase} の層の高さが不正です")
            
            # 層がドメイン内に収まっているか
            if (layer.z_range[0] < 0 or 
                layer.z_range[1] > config.domain.Lz):
                errors.append(f"相 {layer.phase} の層がドメイン外にあります")
        
        # 球の検証
        for sphere in config.initial_condition.spheres:
            # 球の中心がドメイン内にあるか
            for i, (coord, max_val) in enumerate(zip(
                sphere.center, 
                [config.domain.Lx, config.domain.Ly, config.domain.Lz]
            )):
                if coord < 0 or coord > max_val:
                    errors.append(f"球の座標 {i} がドメイン外にあります")
            
            # 球の半径のチェック
            if sphere.radius <= 0:
                errors.append(f"球 {sphere.phase} の半径は正の値である必要があります")
        
        return errors

    @staticmethod
    def _validate_numerical(config: Config) -> List[str]:
        errors = []
        
        # 時間ステップのチェック
        if config.numerical.dt <= 0:
            errors.append("時間刻み幅は正の値である必要があります")
        
        if config.numerical.max_time <= 0:
            errors.append("最大計算時間は正の値である必要があります")
        
        if config.numerical.max_steps <= 0:
            errors.append("最大ステップ数は正の値である必要があります")
        
        if config.numerical.cfl_factor <= 0 or config.numerical.cfl_factor > 1:
            errors.append("CFL係数は0から1の間である必要があります")
        
        if config.numerical.save_interval <= 0:
            errors.append("保存間隔は正の値である必要があります")
        
        return errors

    @staticmethod
    def _validate_boundary_conditions(config: Config) -> List[str]:
        errors = []
        
        # 境界条件の有効な値のリスト
        valid_conditions = ['periodic', 'neumann', 'dirichlet']
        
        for dim, val in [
            ('x', config.boundary_conditions.x),
            ('y', config.boundary_conditions.y),
            ('z', config.boundary_conditions.z)
        ]:
            if val not in valid_conditions:
                errors.append(f"{dim}方向の境界条件は{valid_conditions}のいずれかである必要があります")
        
        return errors