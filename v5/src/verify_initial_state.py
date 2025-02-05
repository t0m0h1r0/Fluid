import numpy as np
from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver.multigrid_poisson_solver import (
    MultigridPoissonSolver as PoissonSolver,
)
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from utils.config import SimulationConfig
from core.boundary import DirectionalBC, NeumannBC, PeriodicBC
import os


def verify_density_field(
    density: np.ndarray,
    config: SimulationConfig,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
):
    """密度場が設定通りになっているか検証する"""
    # 各相の密度を取得
    phase_densities = {phase.name: phase.density for phase in config.phases}
    print("\nPhase densities:")
    for name, density_val in phase_densities.items():
        print(f"  {name}: {density_val} kg/m³")

    # Z座標の生成
    z = np.linspace(0, config.Lz, config.Nz)

    # 各z位置での平均密度をプロファイル化
    z_profile = np.mean(density, axis=(0, 1))
    print("\nDensity profile analysis:")
    print(f"Z coordinates: {z.min():.3f} to {z.max():.3f}")

    # 層構造の検証
    print("\nVerifying layers:")
    for layer in config.layers:
        z_min, z_max = layer.z_range
        expected_density = phase_densities[layer.phase]

        z_indices = np.where((z >= z_min) & (z < z_max))[0]

        if len(z_indices) > 0:
            layer_density = z_profile[z_indices]
            print(f"\nLayer {layer.phase}:")
            print(f"  Z range: [{z_min}, {z_max}]")
            print(f"  Expected density: {expected_density}")
            print(
                f"  Actual density range: [{layer_density.min():.2f}, {layer_density.max():.2f}]"
            )
            print(f"  Mean density: {layer_density.mean():.2f}")
            print(f"  Points in layer: {len(z_indices)}")

            density_diff = np.abs(layer_density - expected_density)
            if np.max(density_diff) > 1e-6:
                print(
                    f"  WARNING: Maximum density deviation: {np.max(density_diff):.2e}"
                )
        else:
            print(
                f"\nWARNING: No points found in layer {layer.phase} at z=[{z_min}, {z_max}]"
            )

    # スフィアの検証
    print("\nVerifying spheres:")
    for sphere in config.spheres:
        expected_density = phase_densities[sphere.phase]
        r = np.sqrt(
            (X - sphere.center[0]) ** 2
            + (Y - sphere.center[1]) ** 2
            + (Z - sphere.center[2]) ** 2
        )
        mask = r <= sphere.radius

        if np.sum(mask) > 0:
            sphere_density = density[mask]
            print(f"\nSphere {sphere.phase}:")
            print(f"  Center: {sphere.center}")
            print(f"  Radius: {sphere.radius}")
            print(f"  Expected density: {expected_density}")
            print(
                f"  Actual density range: [{sphere_density.min():.2f}, {sphere_density.max():.2f}]"
            )
            print(f"  Mean density: {sphere_density.mean():.2f}")
            print(f"  Points in sphere: {np.sum(mask)}")

            density_diff = np.abs(sphere_density - expected_density)
            if np.max(density_diff) > 1e-6:
                print(
                    f"  WARNING: Maximum density deviation: {np.max(density_diff):.2e}"
                )
        else:
            print(
                f"\nWARNING: No points found in sphere at {sphere.center} with radius {sphere.radius}"
            )


def main():
    # 設定の読み込みとソルバーの初期化
    print("Loading configuration and initializing solvers...")
    config = SimulationConfig("config/simulation.yaml")

    scheme = CompactScheme()
    boundary_conditions = DirectionalBC(
        x_bc=PeriodicBC(), y_bc=PeriodicBC(), z_bc=NeumannBC()
    )

    poisson_solver = PoissonSolver(
        scheme=scheme,
        boundary_conditions=[boundary_conditions.get_condition(i) for i in range(3)],
    )

    fluids = {
        phase.name: FluidProperties(phase.name, phase.density, phase.viscosity)
        for phase in config.phases
    }
    fluid_properties = MultiPhaseProperties(fluids)

    phase_solver = PhaseFieldSolver(
        scheme=scheme,
        boundary_conditions=boundary_conditions,
        params=PhaseFieldParams(),
    )

    ns_solver = NavierStokesSolver(
        scheme=scheme,
        boundary_conditions=boundary_conditions,
        poisson_solver=poisson_solver,
        fluid_properties=fluid_properties,
    )

    # フィールドの初期化
    print("\nInitializing fields...")
    velocity = [np.zeros((config.Nx, config.Ny, config.Nz)) for _ in range(3)]
    pressure = np.zeros((config.Nx, config.Ny, config.Nz))

    x = np.linspace(0, config.Lx, config.Nx)
    y = np.linspace(0, config.Ly, config.Ny)
    z = np.linspace(0, config.Lz, config.Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    density = np.zeros((config.Nx, config.Ny, config.Nz))

    # まずレイヤーを設定
    print("\nSetting up layers...")
    for layer in config.layers:
        z_min, z_max = layer.z_range
        phase_density = next(p.density for p in config.phases if p.name == layer.phase)
        mask = (Z >= z_min) & (Z < z_max)
        density[mask] = phase_density
        print(f"Layer {layer.phase} set: z=[{z_min}, {z_max}], density={phase_density}")

    # 次にスフィアを設定（レイヤーを上書きする）
    print("\nSetting up spheres...")
    for sphere in config.spheres:
        phase_density = next(p.density for p in config.phases if p.name == sphere.phase)
        r = np.sqrt(
            (X - sphere.center[0]) ** 2
            + (Y - sphere.center[1]) ** 2
            + (Z - sphere.center[2]) ** 2
        )
        mask = r <= sphere.radius
        density[mask] = phase_density
        print(
            f"Sphere {sphere.phase} set: center={sphere.center}, radius={sphere.radius}, density={phase_density}"
        )

    # 密度場の検証
    print("\nVerifying density field...")
    verify_density_field(density, config, X, Y, Z)

    # 結果の保存
    print("\nSaving initial state visualization...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    writer = DataWriter(output_dir)
    writer.save_density_field(density, config, "initial_state_verification")

    print("\nVerification complete!")


if __name__ == "__main__":
    main()
