import sys
import argparse
import yaml
from pathlib import Path
from logger import SimulationLogger, LogConfig
from simulations import SimulationManager, SimulationInitializer, SimulationRunner
from visualization import visualize_simulation_state


def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase flow simulation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict, debug: bool) -> SimulationLogger:
    log_level = "debug" if debug else config.get("debug", {}).get("level", "info")
    log_dir = Path(config.get("visualization", {}).get("output_dir", "results"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return SimulationLogger("TwoPhaseFlow", LogConfig(level=log_level, log_dir=log_dir))


def initialize_simulation(
    config: dict, logger: SimulationLogger, checkpoint: Path = None
):
    output_dir = Path(config["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        runner, state = SimulationRunner.from_checkpoint(checkpoint, config, logger)
    else:
        logger.info("Starting new simulation")
        initializer = SimulationInitializer(config, logger)
        manager = SimulationManager(config, logger)
        state = initializer.create_initial_state()
        runner = manager.runner
        runner.initialize(state)
        visualize_simulation_state(state, config, timestamp=0.0)

    return runner, state


def run_simulation_loop(runner, config, logger):
    save_interval = config["numerical"].get("save_interval", 0.1)
    max_time = config["numerical"].get("max_time", 1.0)
    output_dir = Path(config["visualization"]["output_dir"])

    next_save_time = save_interval

    while True:
        status = runner.get_status()
        current_time = status["current_time"]

        if current_time >= max_time:
            break

        state, step_info = runner.step_forward()

        if current_time >= next_save_time:
            checkpoint_path = output_dir / f"checkpoint_{current_time:.3f}.npz"
            runner.save_checkpoint(checkpoint_path)
            next_save_time += save_interval

        visualize_simulation_state(state, config, timestamp=current_time)
        logger.info(
            f"Step {step_info['step']}: t={current_time:.3f}, dt={step_info['dt']:.3e}"
        )

    runner.finalize(output_dir)
    logger.info("Simulation completed successfully")


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config, args.debug)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    runner, _ = initialize_simulation(config, logger, checkpoint)
    sys.exit(0)
    run_simulation_loop(runner, config, logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
