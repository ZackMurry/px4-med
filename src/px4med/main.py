"""Entry point. Parses config, waits for containers, runs experiment loop, writes logs."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml


def main() -> None:
    """CLI entry point (registered as `px4med` in pyproject.toml)."""
    asyncio.run(_async_main())


async def _async_main() -> None:
    parser = argparse.ArgumentParser(description="Run px4med MARL experiment")
    parser.add_argument(
        "--config", type=Path, default=Path("config/experiment.yaml"),
        help="Path to experiment YAML config (default: config/experiment.yaml)",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/agent_marl9.pth"),
        help="Path to trained .pth policy file",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for jsonlines output (default: logs/)",
    )
    parser.add_argument(
        "--no-docker", action="store_true",
        help="Skip Docker container lifecycle (assumes SITL already running)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=800,
        help="Maximum steps per episode (default: 800)",
    )
    parser.add_argument(
        "--speed-factor", type=float, default=3.0,
        help="Scale PX4 cruise/max horizontal and vertical speed limits (default: 3.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("px4med.main")

    # Load config
    config: dict = {}
    if args.config.exists():
        with args.config.open() as f:
            config = yaml.safe_load(f) or {}
        logger.info("Config loaded from %s", args.config)
    else:
        logger.warning("Config file not found: %s — using defaults", args.config)

    # Graceful shutdown on SIGINT / SIGTERM
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown.set)
    loop.add_signal_handler(signal.SIGTERM, shutdown.set)

    # Late imports so mavsdk is only required at runtime
    from .coordinator import Coordinator
    from .docker_manager import DockerManager
    from .drone import Drone
    from .environment import WorldEnvironment
    from .metrics import EpisodeRecord, MetricsCollector
    from .policy import PolicyNet

    dm: Optional[DockerManager] = None
    drones: list[Drone] = []
    metrics: Optional[MetricsCollector] = None

    try:
        # ── Docker lifecycle ──────────────────────────────────────────────
        if not args.no_docker:
            dm = DockerManager(log_dir=args.log_dir)
            logger.info("Starting PX4 SITL container ...")
            dm.start()
            logger.info("Waiting for both drones to be ready ...")
            await dm.wait_healthy()
            logger.info("SITL healthy.")

        # ── Connect drones ────────────────────────────────────────────────
        addresses = dm.mavsdk_addresses if dm else [
            f"udpin://0.0.0.0:{14540 + i}" for i in range(2)
        ]
        drones = [Drone(i, addresses[i], grpc_port=50051 + i) for i in range(len(addresses))]
        logger.info("Connecting to %d drone(s) ...", len(drones))
        # Connect sequentially to avoid concurrent MAVSDK backend startup races.
        for d in drones:
            await d.connect()
        logger.info("All drones connected.")
        if args.speed_factor != 1.0:
            logger.info("Applying PX4 speed factor %.2f ...", args.speed_factor)
        for d in drones:
            await d.configure_speed_profile(args.speed_factor)

        # ── Load policy ───────────────────────────────────────────────────
        policy = PolicyNet(args.model)
        logger.info("Policy loaded from %s (device=%s)", args.model, policy.device)

        # ── World + metrics ───────────────────────────────────────────────
        world = WorldEnvironment(config)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = MetricsCollector(args.log_dir)
        metrics.open(run_id)
        logger.info("Run ID: %s — logs → %s", run_id, args.log_dir)

        coordinator = Coordinator(drones, policy, world, metrics)

        # ── Episode loop ──────────────────────────────────────────────────
        for ep in range(args.episodes):
            if shutdown.is_set():
                logger.info("Shutdown requested — stopping before episode %d", ep)
                break

            logger.info("═══ Episode %d / %d ═══", ep + 1, args.episodes)
            summary = await coordinator.run_episode(
                episode=ep, max_steps=args.max_steps
            )

            metrics.log_episode(EpisodeRecord(
                episode=ep,
                steps=summary["steps"],
                patients_delivered=summary["patients_delivered"],
                patients_died=summary["patients_died"],
                patients_spawned=summary["patients_spawned"],
                both_landed=summary["both_landed"],
                battery_remaining=summary["battery_remaining"],
                simulated_battery_remaining=summary["simulated_battery_remaining"],
                total_reward=summary["total_reward"],
                triage_efficiency=summary["triage_efficiency"],
                wind_entries=summary["wind_entries"],
                low_signal_entries=summary["low_signal_entries"],
                obstacle_collisions=summary["obstacle_collisions"],
                agent_collisions=summary["agent_collisions"],
            ))
            logger.info(
                "Episode %d result: delivered=%d/%d died=%d eff=%.2f reward=%.2f steps=%d",
                ep,
                summary["patients_delivered"],
                summary["patients_spawned"],
                summary["patients_died"],
                summary["triage_efficiency"],
                summary["total_reward"],
                summary["steps"],
            )

        logger.info("All episodes complete.")

    except Exception:
        logger.exception("Fatal error — attempting graceful shutdown")

    finally:
        # Land all drones before stopping containers
        for i, drone in enumerate(drones):
            try:
                await drone.land()
            except Exception as exc:
                logger.warning("Could not land drone %d: %s", i, exc)

        if metrics is not None:
            metrics.close()
            logger.info("Metrics flushed.")

        if dm is not None:
            logger.info("Stopping Docker container ...")
            dm.stop()


if __name__ == "__main__":
    main()
