#!/usr/bin/env python3
"""Run one offline episode of the learned policy in WorldEnvironment and render it.

Usage:
    poetry run python scripts/visualize_offline_policy.py
    poetry run python scripts/visualize_offline_policy.py --model models/agent_marl9.pth --fps 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from px4med.drone import Telemetry
from px4med.environment import MAX_PATIENT_TIMER, WorldEnvironment
from px4med.policy import PolicyNet
from px4med.state import METERS_PER_CELL, build_state

GRID_SIZE = 50
CELL_SIZE = 18
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
HUD_HEIGHT = 90
AGENT_COLORS = [(200, 0, 0), (0, 0, 200)]
LANDED_COLORS = [(80, 0, 0), (0, 0, 80)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one offline learned-policy episode")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/agent_marl9.pth"),
        help="Path to learned .pth model",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
        help="Maximum episode steps",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Render/update rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260330,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = PolicyNet(args.model)
    world = WorldEnvironment({"grid": {"size": 50, "meters_per_cell": 2.0}})
    world.reset()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + HUD_HEIGHT))
    pygame.display.set_caption("RouteMED Offline Policy Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 16)
    small_font = pygame.font.SysFont("arial", 10)

    total_reward = 0.0
    running = True
    step = 0

    while running and step < args.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        telems = [_telem_from_world(world, i) for i in range(2)]
        states = [
            build_state(i, telems[i], telems[1 - i], world)
            for i in range(2)
        ]
        actions = [
            policy.select_action(states[i], states[1 - i])
            for i in range(2)
        ]

        step_data = world.step(actions)
        total_reward += sum(step_data["rewards"])
        step += 1

        render_world(
            screen=screen,
            world=world,
            step=step,
            max_steps=args.max_steps,
            actions=actions,
            step_data=step_data,
            total_reward=total_reward,
            font=font,
            small_font=small_font,
        )
        clock.tick(args.fps)

        if step_data["done"]:
            break

    print_summary(world, step, total_reward)

    # Keep the final frame visible until the window is closed.
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(10)

    pygame.quit()


def _telem_from_world(world: WorldEnvironment, agent_idx: int) -> Telemetry:
    grid_x, grid_y = world.agent_grids[agent_idx]
    return Telemetry(
        north_m=-grid_y * METERS_PER_CELL,
        east_m=grid_x * METERS_PER_CELL,
        down_m=-5.0,
        battery_pct=world.batteries[agent_idx],
        is_landed=world.landed[agent_idx],
    )


def render_world(
    screen: pygame.Surface,
    world: WorldEnvironment,
    step: int,
    max_steps: int,
    actions: list[int],
    step_data: dict,
    total_reward: float,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    screen.fill((245, 245, 245))

    # Grid background
    grid_rect = pygame.Rect(0, 0, WINDOW_SIZE, WINDOW_SIZE)
    pygame.draw.rect(screen, (255, 255, 255), grid_rect)
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (220, 220, 220), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (220, 220, 220), (0, y), (WINDOW_SIZE, y))

    # Obstacles
    for obs in world.obstacles:
        pygame.draw.rect(
            screen,
            (20, 20, 20),
            pygame.Rect(obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

    # Hazard zones
    for wz in world.wind_zones:
        pygame.draw.rect(
            screen,
            (255, 180, 90),
            pygame.Rect(wz[0] * CELL_SIZE, wz[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )
    for lsz in world.low_signal_zones:
        pygame.draw.rect(
            screen,
            (180, 140, 220),
            pygame.Rect(lsz[0] * CELL_SIZE, lsz[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

    # Landing zones
    zone_colors = [(210, 255, 210), (210, 210, 255)]
    for idx in range(2):
        lz = world.landing_grid(idx)
        pygame.draw.rect(
            screen,
            zone_colors[idx],
            pygame.Rect(lz[0] * CELL_SIZE, lz[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

    # Patients
    for patient in world.patients:
        if not patient.active:
            continue
        px, py = world.patient_grid(patient.idx)
        if patient.delivered:
            color = (180, 180, 180)
        else:
            ratio = patient.timer / MAX_PATIENT_TIMER
            if ratio > 0.6:
                color = (255, 120, 120)
            elif ratio > 0.3:
                color = (255, 205, 70)
            else:
                color = (220, 0, 0)
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )
        weight_surf = small_font.render(f"W:{patient.weight}", True, (30, 30, 30))
        screen.blit(weight_surf, (px * CELL_SIZE + 1, py * CELL_SIZE + 1))
        if not patient.delivered:
            timer_surf = small_font.render(f"T:{patient.timer}", True, (10, 10, 10))
            screen.blit(timer_surf, (px * CELL_SIZE + 1, py * CELL_SIZE + 10))

    # Drones
    for idx, pos in enumerate(world.agent_grids):
        color = LANDED_COLORS[idx] if world.landed[idx] else AGENT_COLORS[idx]
        center = (
            pos[0] * CELL_SIZE + CELL_SIZE // 2,
            pos[1] * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.circle(screen, color, center, CELL_SIZE // 3)
        pygame.draw.circle(screen, (255, 255, 255), center, CELL_SIZE // 3, 2)

    # HUD
    hud_y = WINDOW_SIZE + 8
    remaining = sum(1 for patient in world.patients if patient.active and not patient.delivered)
    deliveries = step_data["deliveries"]
    hud_lines = [
        f"Step {step}/{max_steps}   Actions: {actions}   Deliveries: {deliveries}   Remaining: {remaining}",
        (
            f"Reward(step): {sum(step_data['rewards']):.2f}   Total reward: {total_reward:.2f}   "
            f"Landed: {world.landed}"
        ),
        (
            f"Battery: {[round(b, 1) for b in world.batteries]}   "
            f"Wind entries: {step_data['wind_entries']}   Low-signal entries: {step_data['low_signal_entries']}"
        ),
    ]
    for line in hud_lines:
        surf = font.render(line, True, (20, 20, 20))
        screen.blit(surf, (10, hud_y))
        hud_y += 24

    pygame.display.flip()


def print_summary(world: WorldEnvironment, step: int, total_reward: float) -> None:
    delivered = sum(1 for patient in world.patients if patient.actually_delivered)
    died = sum(1 for patient in world.patients if patient.delivered and not patient.actually_delivered)
    spawned = sum(1 for patient in world.patients if patient.active)
    print(
        f"Episode complete: steps={step} delivered={delivered}/{spawned} "
        f"died={died} landed={world.landed} battery={[round(b, 1) for b in world.batteries]} "
        f"reward={total_reward:.2f}"
    )


if __name__ == "__main__":
    main()
