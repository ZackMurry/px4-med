#!/usr/bin/env python3
"""Generate publication-ready aggregate figures for paper_core_12h_run."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "paper_core_12h_run"
TABLES_DIR = RUN_DIR / "tables"
OUT_DIR = RUN_DIR / "paper_figures"

POLICY_COLORS = {
    "learned": "#1b4965",
    "priority_path": "#d17b0f",
    "nearest_path": "#4c956c",
}
CONDITION_COLOR = "#1b4965"


def load_summary_rows() -> list[dict[str, str]]:
    with (TABLES_DIR / "summary.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_title(title, fontsize=11, weight="semibold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def make_nominal_comparison(rows: list[dict[str, str]]) -> None:
    nominal = [
        row for row in rows
        if row["suite"] == "baseline_comparison" and row["scenario"] == "nominal"
    ]
    order = ["learned", "priority_path", "nearest_path"]
    labels = ["Learned", "Priority", "Nearest"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    metrics = [
        ("delivery_rate_mean", "delivery_rate_ci95", "Delivery Rate", "Rate", (0.0, 1.05)),
        ("triage_efficiency_mean", "triage_efficiency_ci95", "Triage Efficiency", "Score", (0.0, 1.05)),
        ("mission_success_rate_mean", "mission_success_rate_ci95", "Mission Success", "Rate", (0.0, 1.05)),
        ("high_acuity_service_rate_mean", "high_acuity_service_rate_ci95", "High-Acuity Service", "Rate", (0.0, 1.05)),
    ]

    for ax, (metric, ci, title, ylabel, ylim) in zip(axes, metrics, strict=True):
        values = [float(next(r for r in nominal if r["policy"] == p)[metric]) for p in order]
        cis = [float(next(r for r in nominal if r["policy"] == p)[ci]) for p in order]
        colors = [POLICY_COLORS[p] for p in order]
        ax.bar(labels, values, yerr=cis, color=colors, capsize=4)
        _style_axis(ax, title, ylabel, ylim)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Nominal PX4 SITL Policy Comparison", fontsize=13, weight="semibold")
    save(fig, "nominal_policy_comparison.png")


def make_learned_conditions(rows: list[dict[str, str]]) -> None:
    learned = [row for row in rows if row["policy"] == "learned"]
    order = [
        ("baseline_comparison", "nominal", "Nominal"),
        ("battery_sweep", "battery_35", "Battery 35%"),
        ("hazard_sweep", "hazard_high", "Hazard High"),
        ("delay_sweep", "delay_3", "Delay 3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    metrics = [
        ("delivery_rate_mean", "Delivery Rate", "Rate", (0.0, 1.05)),
        ("triage_efficiency_mean", "Triage Efficiency", "Score", (0.0, 1.05)),
        ("mortality_rate_mean", "Mortality Rate", "Rate", (0.0, 0.35)),
        ("mission_success_rate_mean", "Mission Success", "Rate", (0.0, 1.05)),
    ]

    for ax, (metric, title, ylabel, ylim) in zip(axes.ravel(), metrics, strict=True):
        labels: list[str] = []
        values: list[float] = []
        for suite, scenario, label in order:
            row = next(r for r in learned if r["suite"] == suite and r["scenario"] == scenario)
            labels.append(label)
            values.append(float(row[metric]))
        ax.bar(labels, values, color=CONDITION_COLOR)
        _style_axis(ax, title, ylabel, ylim)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Learned RouteMED Policy Across SITL Conditions", fontsize=13, weight="semibold")
    save(fig, "learned_policy_conditions.png")


def make_learned_operational(rows: list[dict[str, str]]) -> None:
    learned = [row for row in rows if row["policy"] == "learned"]
    order = [
        ("baseline_comparison", "nominal", "Nominal"),
        ("battery_sweep", "battery_35", "Battery 35%"),
        ("hazard_sweep", "hazard_high", "Hazard High"),
        ("delay_sweep", "delay_3", "Delay 3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    metrics = [
        ("wrong_land_attempts_mean", "Wrong-Land Attempts", "Count", None),
        ("mean_tracking_error_m_mean", "Mean Tracking Error", "Meters", (0.0, 2.2)),
        ("battery_margin_min_mean", "Minimum Battery Margin", "Percent", (0.0, 25.0)),
        ("steps_mean", "Episode Steps", "Steps", (0.0, 850.0)),
    ]

    for ax, (metric, title, ylabel, ylim) in zip(axes.ravel(), metrics, strict=True):
        labels: list[str] = []
        values: list[float] = []
        for suite, scenario, label in order:
            row = next(r for r in learned if r["suite"] == suite and r["scenario"] == scenario)
            labels.append(label)
            values.append(float(row[metric]))
        ax.bar(labels, values, color=CONDITION_COLOR)
        _style_axis(ax, title, ylabel, ylim)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Operational Characteristics of the Learned Policy", fontsize=13, weight="semibold")
    save(fig, "learned_policy_operational_metrics.png")


def main() -> None:
    rows = load_summary_rows()
    make_nominal_comparison(rows)
    make_learned_conditions(rows)
    make_learned_operational(rows)


if __name__ == "__main__":
    main()
