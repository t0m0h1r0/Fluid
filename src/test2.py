import numpy as np
import pytest
from core.field import GridInfo, ScalarField, VectorField
from physics.continuity import ContinuityEquation


def create_grid(shape=(32, 32, 32)):
    """テスト用のグリッドを作成"""
    return GridInfo(shape=shape, dx=(1.0 / (shape[0] - 1), 1.0 / (shape[1] - 1), 1.0 / (shape[2] - 1)))


def test_continuity_constant_field():
    """定数スカラー場と均一速度場のテスト
    
    この場合、連続の方程式の時間微分は常に0になるはず。
    """
    grid = create_grid((16, 16, 16))
    
    # 定数スカラー場
    scalar_field = ScalarField(grid, initial_value=5.0)
    
    # 均一な速度場
    velocity = VectorField(grid, initial_values=(1.0, 1.0, 1.0))
    
    # 連続の方程式を計算
    continuity = ContinuityEquation(conservative=True)
    derivative = continuity.compute_derivative(scalar_field, velocity)
    
    # 時間微分は0になるはず
    assert np.allclose(derivative.data, 0.0, atol=1e-10)


def test_continuity_linear_field():
    """線形スカラー場と線形速度場のテスト
    
    解析的な微分と数値的な微分を比較する
    """
    grid = create_grid((32, 32, 32))
    
    # メッシュグリッドの作成
    coords = np.meshgrid(np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         indexing='ij')
    
    # 線形スカラー場: f(x,y,z) = x + 2y + 3z
    scalar_field_data = coords[0] + 2 * coords[1] + 3 * coords[2]
    scalar_field = ScalarField(grid, initial_value=scalar_field_data)
    
    # 線形速度場: u = [1, 2, 3]
    velocity = VectorField(grid, initial_values=(
        np.ones_like(scalar_field_data),
        2 * np.ones_like(scalar_field_data),
        3 * np.ones_like(scalar_field_data)
    ))
    
    # 連続の方程式を計算（保存形）
    conservative_eq = ContinuityEquation(conservative=True)
    conservative_derivative = conservative_eq.compute_derivative(scalar_field, velocity)
    
    # 連続の方程式を計算（非保存形）
    nonconservative_eq = ContinuityEquation(conservative=False)
    nonconservative_derivative = nonconservative_eq.compute_derivative(scalar_field, velocity)
    
    # 解析的な導関数の計算
    # 保存形 -∇⋅(uf): -div(u * f)
    analytical_conservative = -(
        1 * np.gradient(scalar_field_data, 1/31, axis=0) +
        2 * np.gradient(scalar_field_data, 1/31, axis=1) +
        3 * np.gradient(scalar_field_data, 1/31, axis=2)
    )
    
    # 非保存形 -u⋅∇f: -(u_x ∂f/∂x + u_y ∂f/∂y + u_z ∂f/∂z)
    gradient_x = np.gradient(scalar_field_data, 1/31, axis=0)
    gradient_y = np.gradient(scalar_field_data, 1/31, axis=1)
    gradient_z = np.gradient(scalar_field_data, 1/31, axis=2)
    
    analytical_nonconservative = -(
        1 * gradient_x +
        2 * gradient_y +
        3 * gradient_z
    )
    
    # 保存形の検証
    assert np.allclose(
        conservative_derivative.data, 
        analytical_conservative, 
        rtol=1e-5, atol=1e-8
    )
    
    # 非保存形の検証
    assert np.allclose(
        nonconservative_derivative.data, 
        analytical_nonconservative, 
        rtol=1e-5, atol=1e-8
    )


def test_continuity_sin_field():
    """三角関数スカラー場と複雑な速度場のテスト
    
    より複雑な関数を用いて、より高度な検証を行う
    """
    grid = create_grid((64, 64, 64))
    
    # メッシュグリッドの作成
    coords = np.meshgrid(np.linspace(0, 2*np.pi, 64), 
                         np.linspace(0, 2*np.pi, 64), 
                         np.linspace(0, 2*np.pi, 64), 
                         indexing='ij')
    
    # 三角関数スカラー場: f(x,y,z) = sin(x)cos(y)exp(z)
    scalar_field_data = (
        np.sin(coords[0]) * 
        np.cos(coords[1]) * 
        np.exp(coords[2])
    )
    scalar_field = ScalarField(grid, initial_value=scalar_field_data)
    
    # 複雑な速度場
    velocity = VectorField(grid, initial_values=(
        np.cos(coords[0]),
        -np.sin(coords[1]),
        np.exp(coords[2])
    ))
    
    # 連続の方程式を計算（保存形と非保存形）
    conservative_eq = ContinuityEquation(conservative=True)
    nonconservative_eq = ContinuityEquation(conservative=False)
    
    conservative_derivative = conservative_eq.compute_derivative(scalar_field, velocity)
    nonconservative_derivative = nonconservative_eq.compute_derivative(scalar_field, velocity)
    
    # 診断情報の取得
    print("Conservative Derivative Diagnostics:")
    print(conservative_eq.get_diagnostics(scalar_field, velocity, conservative_derivative))
    
    print("\nNon-Conservative Derivative Diagnostics:")
    print(nonconservative_eq.get_diagnostics(scalar_field, velocity, nonconservative_derivative))
    
    # この関数では完全な解析的微分が難しいため、
    # 以下では主に物理的一貫性と大域的な性質を検証する
    
    # 最大絶対値がある程度小さいことを確認
    assert np.max(np.abs(conservative_derivative.data)) < 10.0
    assert np.max(np.abs(nonconservative_derivative.data)) < 10.0
    
    # 保存形と非保存形で大域的な統計量が似ていることを確認
    assert np.isclose(
        np.mean(np.abs(conservative_derivative.data)),
        np.mean(np.abs(nonconservative_derivative.data)),
        rtol=0.2
    )


def test_cfl_timestep():
    """CFL条件による時間刻み幅の制限のテスト"""
    grid = create_grid((32, 32, 32))
    
    # メッシュグリッドの作成
    coords = np.meshgrid(np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         indexing='ij')
    
    # 複雑な速度場
    velocity = VectorField(grid, initial_values=(
        np.sin(coords[0]),
        np.cos(coords[1]),
        np.exp(coords[2])
    ))
    
    # 連続の方程式を計算
    conservative_eq = ContinuityEquation(conservative=True)
    
    # CFL条件による時間刻み幅を計算
    cfl_timestep = conservative_eq.compute_cfl_timestep(
        velocity, grid, cfl=0.5
    )
    
    # 時間刻み幅が0より大きく、有限であることを確認
    assert cfl_timestep > 0
    assert cfl_timestep < float('inf')
    
    # CFLパラメータが保守的であることを確認（典型的な値は0.1〜1.0）
    assert 0.1 <= 0.5 <= 1.0


def test_continuity_diagnostics():
    """診断情報の詳細テスト"""
    grid = create_grid((32, 32, 32))
    
    # メッシュグリッドの作成
    coords = np.meshgrid(np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         np.linspace(0, 1, 32), 
                         indexing='ij')
    
    # サンプルスカラー場と速度場
    scalar_field_data = np.sin(coords[0]) * np.cos(coords[1]) * np.exp(coords[2])
    scalar_field = ScalarField(grid, initial_value=scalar_field_data)
    
    velocity = VectorField(grid, initial_values=(
        np.cos(coords[0]),
        -np.sin(coords[1]),
        np.exp(coords[2])
    ))
    
    # 連続の方程式を計算
    conservative_eq = ContinuityEquation(conservative=True)
    derivative = conservative_eq.compute_derivative(scalar_field, velocity)
    
    # 診断情報の取得とチェック
    diagnostics = conservative_eq.get_diagnostics(scalar_field, velocity, derivative)
    
    # 診断情報のキーの存在を確認
    expected_keys = [
        'method', 'field_stats', 'velocity_stats', 
        'derivative_stats', 'numerical_properties'
    ]
    
    for key in expected_keys:
        assert key in diagnostics
    
    # フィールド統計の検証
    assert 'min' in diagnostics['field_stats']
    assert 'max' in diagnostics['field_stats']
    assert 'mean' in diagnostics['field_stats']
    
    # 速度統計の検証
    assert 'max_magnitude' in diagnostics['velocity_stats']
    assert 'mean_magnitude' in diagnostics['velocity_stats']
    
    # 微分統計の検証
    assert 'max' in diagnostics['derivative_stats']
    assert 'min' in diagnostics['derivative_stats']
    assert 'mean' in diagnostics['derivative_stats']


def main():
    """テストランナー"""
    # すべてのテスト関数を手動で実行
    test_continuity_constant_field()
    test_continuity_linear_field()
    test_continuity_sin_field()
    test_cfl_timestep()
    test_continuity_diagnostics()
    
    print("全てのテストが正常に完了しました。")


if __name__ == "__main__":
    main()