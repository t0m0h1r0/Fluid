from pathlib import Path
from typing import Dict, List
from .base import Config

class ConfigValidator:
    """設定の妥当性検証"""
    
    @staticmethod
    def validate(config: Config) -> List[str]:
        errors = []
        
        # ドメイン設定の検証
        errors.extend(ConfigValidator._validate_domain(config))
        
        # 流体設定の検証
        errors.extend(ConfigValidator._validate_fluids(config))
        
        # 相場設定の検証
        errors.extend(ConfigValidator._validate_phase(config))
        
        # 数値計算設定の検証
        errors.extend(ConfigValidator._validate_numerical(config))
        
        # 出力設定の検証
        errors.extend(ConfigValidator._validate_output(config))
        
        return errors

    @staticmethod
    def _validate_domain(config: Config) -> List[str]:
        errors = []
        
        # サイズの検証
        if any(s <= 0 for s in config.domain.size):
            errors.append("ドメインサイズは正の値である必要があります")
        
        # 解像度の検証
        if any(n <= 0 for n in config.domain.resolution):
            errors.append("解像度は正の値である必要があります")
        
        # グリッド数の制限
        max_points = 1e7  # 1000万点
        total_points = (config.domain.resolution[0] * 
                       config.domain.resolution[1] * 
                       config.domain.resolution[2])
        if total_points > max_points:
            errors.append(f"総グリッド数が多すぎます: {total_points} > {max_points}")
        
        return errors

    @staticmethod
    def _validate_fluids(config: Config) -> List[str]:
        errors = []
        
        if not config.fluids:
            errors.append("流体が定義されていません")
            return errors
        
        for name, fluid in config.fluids.items():
            # 物性値の範囲チェック
            if fluid.density <= 0:
                errors.append(f"流体 {name} の密度は正の値である必要があります")
            
            if fluid.viscosity <= 0:
                errors.append(f"流体 {name} の粘性係数は正の値である必要があります")
            
            if fluid.surface_tension is not None and fluid.surface_tension < 0:
                errors.append(f"流体 {name} の表面張力係数は非負である必要があります")
            
            if fluid.specific_heat is not None and fluid.specific_heat <= 0:
                errors.append(f"流体 {name} の比熱は正の値である必要があります")
            
            if (fluid.thermal_conductivity is not None and 
                fluid.thermal_conductivity <= 0):
                errors.append(f"流体 {name} の熱伝導率は正の値である必要があります")
        
        return errors

    @staticmethod
    def _validate_phase(config: Config) -> List[str]:
        errors = []
        
        # 界面パラメータの検証
        if config.phase.epsilon <= 0:
            errors.append("界面厚さは正の値である必要があります")
        
        if config.phase.mobility <= 0:
            errors.append("移動度は正の値である必要があります")
        
        if config.phase.surface_tension < 0:
            errors.append("表面張力係数は非負である必要があります")
        
        if config.phase.stabilization < 0:
            errors.append("安定化係数は非負である必要があります")
        
        return errors

    @staticmethod
    def _validate_numerical(config: Config) -> List[str]:
        errors = []
        
        # 時間ステップの検証
        if config.numerical.dt <= 0:
            errors.append("時間刻み幅は正の値である必要があります")
        
        if config.numerical.max_time <= 0:
            errors.append("最大計算時間は正の値である必要があります")
        
        if config.numerical.cfl_number <= 0 or config.numerical.cfl_number > 1:
            errors.append("CFL数は0から1の間である必要があります")
        
        if config.numerical.tolerance <= 0:
            errors.append("収束判定閾値は正の値である必要があります")
        
        if config.numerical.max_iterations <= 0:
            errors.append("最大反復回数は正の値である必要があります")
        
        return errors

    @staticmethod
    def _validate_output(config: Config) -> List[str]:
        errors = []
        
        # 保存間隔の検証
        if config.output.save_interval <= 0:
            errors.append("保存間隔は正の値である必要があります")
        
        # 保存形式の検証
        valid_formats = ['hdf5', 'csv']
        if config.output.save_format not in valid_formats:
            errors.append(f"保存形式は {valid_formats} のいずれかである必要があります")
        
        # 出力ディレクトリの検証
        if not config.output.output_dir.parent.exists():
            errors.append("出力ディレクトリの親ディレクトリが存在しません")
        
        return errors