#!/usr/bin/env python3
"""Publication figures + analysis tables for a px4-med validation run.

Reads a run directory's `tables/{episodes,summary}.csv` (written incrementally
by scripts/run_overnight_validation.py) and emits, into `<run>/paper_figures/`:

  fig1_nominal_comparison.png   delivery / triage / mission success / all-landed
  fig2_hazard_discipline.png    wind + low-signal avoidance rates (with the
                                opportunity denominators that make the claim)
  fig3_energy_discipline.png    reserve violations / depletions / min battery
  fig4_triage_by_acuity.png     per-acuity delivery rate + response time
  fig5_transfer_offline_vs_sitl.png   the sim-transfer gap (needs --offline-dir)
  fig6_battery_sweep.png        nominal vs battery_60
  analysis.md                   headline numbers in prose + markdown tables
  table_main.tex                booktabs table, mean +- CI95

Safe to run against a partially complete run: missing policies, scenarios or
metrics are skipped with a note rather than raising, so this can be called
while the suite is still going.

Usage:
  poetry run python scripts/make_paper_figures.py --run-dir results/core_...
  poetry run python scripts/make_paper_figures.py --run-dir R --offline-dir O
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

# ── palette ───────────────────────────────────────────────────────────────────
# Categorical slots 1-4 of the validated default palette, assigned in fixed
# order by policy identity (never cycled, never by rank). Verified with the
# dataviz validator: all checks pass on the adjacent pairlist; the contrast WARN
# on aqua/yellow is relieved by the direct value labels every bar carries.
POLICY_ORDER = ("learned", "priority_path", "nearest_path", "random")
POLICY_COLORS = {
    "learned": "#2a78d6",        # slot 1 blue
    "priority_path": "#eb6834",  # slot 2 orange
    "nearest_path": "#1baf7a",   # slot 3 aqua
    "random": "#eda100",         # slot 4 yellow
}
# Secondary encoding so identity survives greyscale printing and CVD.
POLICY_HATCH = {
    "learned": "",
    "priority_path": "///",
    "nearest_path": "...",
    "random": "xxx",
}
POLICY_LABELS = {
    "learned": "Learned\n(CEDA-FGCS)",
    "priority_path": "Priority\npath",
    "nearest_path": "Nearest\npath",
    "random": "Random",
}
# Pooled nominal (zero-delay) reference from core_20260829_230535 +
# extend_20260831_000000, learned policy, n=19. Used as the delay=0 anchor in
# the latency figure because the latency suite has no delay_0 scenario.
ZERO_DELAY_REFERENCE = {
    "delivery_rate": (0.701, 0.032),
    "triage_efficiency": (0.715, 0.040),
    "drones_landed": (4.158, 0.456),
    "wind_avoidance_rate": (0.998, 0.002),
}

ACUITY_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

INK = "#1a1a1a"
MUTED = "#5c5c5c"
GRID = "#d8d8d8"


# ── data loading ──────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: Optional[dict[str, str]], key: str) -> Optional[float]:
    """Float or None — missing row/blank/non-numeric all degrade to None."""
    if row is None:
        return None
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def pick(rows: list[dict[str, str]], **criteria) -> list[dict[str, str]]:
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def summary_row(
    rows: list[dict[str, str]], suite: str, scenario: str, policy: str
) -> Optional[dict[str, str]]:
    found = pick(rows, suite=suite, scenario=scenario, policy=policy)
    return found[0] if found else None


# ── plot helpers ──────────────────────────────────────────────────────────────

def style_axis(ax, title: str, ylabel: str, ylim=None) -> None:
    ax.set_title(title, fontsize=11, weight="semibold", color=INK, pad=9)
    ax.set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9, colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if ylim is not None:
        ax.set_ylim(*ylim)


def label_bars(ax, bars, values, fmt: str = "{:.2f}", errors=None) -> None:
    """Direct value labels — also the relief for the palette's contrast WARN.

    Labels sit above the error-bar whisker, not the bar top, or they collide
    with the cap.
    """
    errors = list(errors) if errors is not None else [0.0] * len(bars)
    for bar, value, error in zip(bars, values, errors):
        if value is None:
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (error or 0.0)),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=8.5, color=INK,
        )


def policy_bars(
    ax, rows, suite, scenario, metric, *, policies=POLICY_ORDER, fmt="{:.2f}"
) -> bool:
    """One panel of per-policy bars with CI95. False if no data at all."""
    present, values, errors, colors, hatches, labels = [], [], [], [], [], []
    for policy in policies:
        row = summary_row(rows, suite, scenario, policy)
        if row is None:
            continue
        value = fnum(row, f"{metric}_mean")
        if value is None:
            continue
        present.append(policy)
        values.append(value)
        errors.append(fnum(row, f"{metric}_ci95") or 0.0)
        colors.append(POLICY_COLORS[policy])
        hatches.append(POLICY_HATCH[policy])
        labels.append(POLICY_LABELS[policy])
    if not present:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=MUTED)
        return False

    bars = ax.bar(
        labels, values, yerr=errors, color=colors, capsize=3.5,
        width=0.62, error_kw={"ecolor": MUTED, "elinewidth": 1.1},
        edgecolor="white", linewidth=1.4,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    label_bars(ax, bars, values, fmt, errors)
    return True


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── figures ───────────────────────────────────────────────────────────────────

def fig_nominal(rows, out_dir: Path) -> Optional[Path]:
    panels = [
        ("delivery_rate", "Delivery rate", "fraction of patients", (0, 1.12), "{:.2f}"),
        ("triage_efficiency", "Triage efficiency", "acuity-weighted", (0, 1.12), "{:.2f}"),
        ("mission_success", "Mission success", "fraction of episodes", (0, 1.12), "{:.2f}"),
        ("all_landed", "All drones landed", "fraction of episodes", (0, 1.12), "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 3.5))
    any_data = False
    for ax, (metric, title, ylabel, ylim, fmt) in zip(axes, panels):
        any_data |= policy_bars(
            ax, rows, "baseline_comparison", "nominal", metric, fmt=fmt)
        style_axis(ax, title, ylabel, ylim)
    fig.suptitle(
        "PX4 SITL, nominal scenario — mean over episodes, error bars 95% CI",
        fontsize=10.5, color=MUTED, y=1.04,
    )
    return save(fig, out_dir, "fig1_nominal_comparison.png") if any_data else None


def fig_hazard(rows, out_dir: Path) -> Optional[Path]:
    """Avoidance rates, annotated with the opportunity counts.

    A rate alone is unfalsifiable here: the env scores a vacuous 1.0 when a
    policy never had the chance to enter a hazard, so the denominator is
    printed under every bar.
    """
    panels = [
        ("wind_avoidance_rate", "wind_avoidance_opportunities",
         "Wind avoidance", "1 − selections / opportunities"),
        ("low_signal_avoidance_rate", "low_signal_avoidance_opportunities",
         "Low-signal avoidance", "1 − selections / opportunities"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    any_data = False
    for ax, (metric, denom, title, ylabel) in zip(axes, panels):
        ok = policy_bars(ax, rows, "baseline_comparison", "nominal", metric)
        any_data |= ok
        style_axis(ax, title, ylabel, (0, 1.18))
        if ok:
            notes = []
            for policy in POLICY_ORDER:
                row = summary_row(rows, "baseline_comparison", "nominal", policy)
                if row is None or fnum(row, f"{metric}_mean") is None:
                    continue
                opportunities = fnum(row, f"{denom}_mean")
                notes.append(
                    "n/a" if opportunities is None else f"{opportunities:.0f}"
                )
            ax.set_xticks(range(len(notes)))
            ax.set_xticklabels(
                [
                    f"{ax.get_xticklabels()[i].get_text()}\n({notes[i]} opp.)"
                    for i in range(len(notes))
                ],
                fontsize=9, color=MUTED,
            )
        # Stage-2 training gate: the bar the policy was trained to clear.
        # Annotated below-left so it clears the value labels sitting on top of
        # near-1.0 bars.
        ax.axhline(0.98, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            "training gate 0.98", (0.02, 0.965),
            xycoords=("axes fraction", "data"),
            ha="left", va="top", fontsize=8, color=MUTED,
        )
    fig.suptitle(
        "Hazard discipline — mean per-episode avoidance rate, 95% CI "
        "(opportunity counts in parentheses)",
        fontsize=10.5, color=MUTED, y=1.04,
    )
    return save(fig, out_dir, "fig2_hazard_discipline.png") if any_data else None


def fig_energy(rows, out_dir: Path) -> Optional[Path]:
    panels = [
        ("reserve_violations", "Energy-reserve violations",
         "steps below reserve", None, "{:.0f}"),
        ("drones_depleted", "Drones depleted", "count of 5", (0, 5.4), "{:.2f}"),
        ("sim_battery_min", "Worst final battery", "% remaining", (0, 105), "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(10.5, 3.6))
    any_data = False
    for ax, (metric, title, ylabel, ylim, fmt) in zip(axes, panels):
        any_data |= policy_bars(
            ax, rows, "baseline_comparison", "nominal", metric, fmt=fmt)
        style_axis(ax, title, ylabel, ylim)
    fig.suptitle(
        "Mission-energy discipline — nominal scenario, 95% CI",
        fontsize=10.5, color=MUTED, y=1.04,
    )
    return save(fig, out_dir, "fig3_energy_discipline.png") if any_data else None


def fig_triage_by_acuity(rows, out_dir: Path) -> Optional[Path]:
    """Grouped by acuity class; series = policy. Two measures -> two panels."""
    groups = [
        ("delivery_rate_w{}", "Delivery rate by acuity", "fraction", (0, 1.15), "{:.2f}"),
        ("mean_response_time_w{}", "Response time by acuity", "steps", None, "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    any_data = False
    for ax, (template, title, ylabel, ylim, fmt) in zip(axes, groups):
        policies = [
            p for p in POLICY_ORDER
            if summary_row(rows, "baseline_comparison", "nominal", p) is not None
        ]
        if not policies:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=MUTED)
            style_axis(ax, title, ylabel, ylim)
            continue
        width = 0.8 / len(policies)
        for offset, policy in enumerate(policies):
            row = summary_row(rows, "baseline_comparison", "nominal", policy)
            values, errors = [], []
            for weight in (1, 2, 3):
                metric = template.format(weight)
                values.append(fnum(row, f"{metric}_mean") or 0.0)
                errors.append(fnum(row, f"{metric}_ci95") or 0.0)
            positions = [i + offset * width - 0.4 + width / 2 for i in range(3)]
            bars = ax.bar(
                positions, values, width=width * 0.92, yerr=errors,
                color=POLICY_COLORS[policy], capsize=2.5,
                error_kw={"ecolor": MUTED, "elinewidth": 0.9},
                edgecolor="white", linewidth=1.0,
                label=POLICY_LABELS[policy].replace("\n", " "),
                hatch=POLICY_HATCH[policy],
            )
            any_data = True
            if len(policies) <= 4:
                label_bars(ax, bars, values, fmt, errors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(
            ["W1 (minor)", "W2 (delayed)", "W3 (immediate)"],
            fontsize=9, color=MUTED,
        )
        style_axis(ax, title, ylabel, ylim)
        ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(
        "Triage behaviour by acuity class — W3 is most urgent; 95% CI",
        fontsize=10.5, color=MUTED, y=1.03,
    )
    return save(fig, out_dir, "fig4_triage_by_acuity.png") if any_data else None


def fig_transfer(sitl_rows, offline_rows, out_dir: Path) -> Optional[Path]:
    """Offline (abstract env) vs SITL (PX4) for the learned policy."""
    if not offline_rows:
        return None
    metrics = [
        ("delivery_rate", "Delivery rate", "fraction", (0, 1.15), "{:.2f}"),
        ("triage_efficiency", "Triage efficiency", "acuity-weighted", (0, 1.15), "{:.2f}"),
        ("all_landed", "All drones landed", "fraction of episodes", (0, 1.15), "{:.2f}"),
        ("steps", "Episode length", "steps", None, "{:.0f}"),
    ]
    backends = [
        ("offline", "Abstract env\n(offline)", "#2a78d6", ""),
        ("sitl", "PX4 SITL", "#eb6834", "///"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.0, 3.6))
    any_data = False
    for ax, (metric, title, ylabel, ylim, fmt) in zip(axes, metrics):
        values, errors, colors, hatches, labels = [], [], [], [], []
        for key, label, color, hatch in backends:
            source = offline_rows if key == "offline" else sitl_rows
            row = summary_row(source, "baseline_comparison", "nominal", "learned")
            if row is None:
                continue
            value = fnum(row, f"{metric}_mean")
            if value is None:
                continue
            values.append(value)
            errors.append(fnum(row, f"{metric}_ci95") or 0.0)
            colors.append(color)
            hatches.append(hatch)
            labels.append(label)
        if not values:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=MUTED)
            style_axis(ax, title, ylabel, ylim)
            continue
        bars = ax.bar(
            labels, values, yerr=errors, color=colors, capsize=3.5, width=0.55,
            error_kw={"ecolor": MUTED, "elinewidth": 1.1},
            edgecolor="white", linewidth=1.4,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        label_bars(ax, bars, values, fmt, errors)
        any_data = True
        style_axis(ax, title, ylabel, ylim)
    fig.suptitle(
        "Sim-to-sim transfer for the learned policy — identical world model, "
        "abstract stepping vs PX4 flight; 95% CI",
        fontsize=10.5, color=MUTED, y=1.04,
    )
    return save(fig, out_dir, "fig5_transfer_offline_vs_sitl.png") if any_data else None


def fig_battery_sweep(rows, out_dir: Path) -> Optional[Path]:
    metrics = [
        ("delivery_rate", "Delivery rate", "fraction", (0, 1.15)),
        ("triage_efficiency", "Triage efficiency", "acuity-weighted", (0, 1.15)),
        ("drones_depleted", "Drones depleted", "count of 5", (0, 5.4)),
    ]
    conditions = [
        ("baseline_comparison", "nominal", "100% battery"),
        ("battery_sweep", "battery_60", "60% battery"),
    ]
    policies = ("learned", "priority_path")
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.0, 3.7))
    any_data = False
    for ax, (metric, title, ylabel, ylim) in zip(axes, metrics):
        width = 0.35
        for offset, policy in enumerate(policies):
            values, errors, positions = [], [], []
            for i, (suite, scenario, _) in enumerate(conditions):
                row = summary_row(rows, suite, scenario, policy)
                value = fnum(row, f"{metric}_mean") if row else None
                values.append(value or 0.0)
                errors.append((fnum(row, f"{metric}_ci95") or 0.0) if row else 0.0)
                positions.append(i + offset * width - width / 2)
                if value is not None:
                    any_data = True
            bars = ax.bar(
                positions, values, width=width * 0.9, yerr=errors,
                color=POLICY_COLORS[policy], capsize=2.5,
                error_kw={"ecolor": MUTED, "elinewidth": 0.9},
                edgecolor="white", linewidth=1.0,
                hatch=POLICY_HATCH[policy],
                label=POLICY_LABELS[policy].replace("\n", " "),
            )
            label_bars(ax, bars, values, errors=errors)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c[2] for c in conditions], fontsize=9, color=MUTED)
        style_axis(ax, title, ylabel, ylim)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.suptitle(
        "Battery stress — starting charge reduced to 60%; 95% CI",
        fontsize=10.5, color=MUTED, y=1.04,
    )
    return save(fig, out_dir, "fig6_battery_sweep.png") if any_data else None


def fig_trajectories(run_dir: Path, out_dir: Path) -> Optional[Path]:
    """Qualitative panel: the five drones' paths for one learned episode.

    Read straight from steps.csv, so it shows the actual flown grid track
    (telemetry-synced), not a replay. Hazard rectangles are deliberately not
    overlaid: their geometry is not in steps.csv and re-deriving it from a
    fresh env would not be guaranteed to reproduce this episode's layout.
    """
    steps = read_csv(run_dir / "tables" / "steps.csv")
    if not steps:
        return None
    learned = [
        r for r in steps
        if r.get("policy") == "learned" and r.get("scenario") == "nominal"
    ]
    if not learned:
        return None
    # One episode: whichever learned episode has the most recorded steps.
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in learned:
        by_episode.setdefault(row.get("episode", "0"), []).append(row)
    episode_key = max(by_episode, key=lambda k: len(by_episode[k]))
    rows = sorted(by_episode[episode_key], key=lambda r: int(r["step"]))

    tracks: list[list[tuple[int, int]]] = []
    deliveries: list[tuple[int, int]] = []
    for row in rows:
        cells = []
        for token in (row.get("positions") or "").split(";"):
            if ":" not in token:
                continue
            x, y = token.split(":", 1)
            try:
                cells.append((int(x), int(y)))
            except ValueError:
                continue
        if not tracks:
            tracks = [[] for _ in cells]
        for i, cell_xy in enumerate(cells):
            if i < len(tracks):
                tracks[i].append(cell_xy)
        if row.get("deliveries"):
            for i, cell_xy in enumerate(cells):
                if i == 0:
                    deliveries.append(cell_xy)
    if not tracks or not any(tracks):
        return None

    # Slot 1-5 hues: drone identity, fixed order.
    drone_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    for i, track in enumerate(tracks):
        if not track:
            continue
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        color = drone_colors[i % len(drone_colors)]
        ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.85,
                label=f"Drone {i}", solid_capstyle="round")
        ax.plot(xs[0], ys[0], marker="o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.plot(xs[-1], ys[-1], marker="s", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4)

    ax.set_xlim(-2, 101)
    ax.set_ylim(101, -2)          # grid y grows southward
    ax.set_aspect("equal")
    ax.set_xlabel("grid x  (2 m per cell)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("grid y  (2 m per cell)", fontsize=9.5, color=MUTED)
    backend = rows[0].get("backend", "")
    backend_label = {
        "sitl": "PX4 SITL", "offline": "abstract env (offline)",
    }.get(backend, backend or "unknown backend")
    ax.set_title(
        f"Learned policy, {backend_label}, episode {episode_key} — "
        f"{len(rows)} steps\ncircle = start, square = final position",
        fontsize=10.5, weight="semibold", color=INK, pad=10,
    )
    ax.grid(color=GRID, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(fontsize=8.5, frameon=False, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))
    return save(fig, out_dir, "fig7_trajectories.png")


# ── tables / prose ────────────────────────────────────────────────────────────

def fig_hazard_sweep(rows, out_dir: Path) -> Optional[Path]:
    """Avoidance / exposure / delivery vs hazard density, one line per policy.

    A sweep over an ordered variable is a line chart, not bars. Rates and
    exposure-step counts sit on separate panels because they are different
    measures on different scales (never a dual axis).

    The avoidance panels are annotated with each point's opportunity count,
    because that denominator is NOT monotone in density: an opportunity
    requires both a hazard cell and a hazard-free cell among the valid
    destinations, so it rises and then falls again as the map saturates.
    """
    sweep = [r for r in rows if r.get("suite") == "hazard_sweep"]
    if not sweep:
        return None

    def density(row) -> float:
        try:
            return int(row["scenario"].split("_")[-1]) / 100.0
        except (ValueError, KeyError, IndexError):
            return float("nan")

    policies = [p for p in POLICY_ORDER if any(r["policy"] == p for r in sweep)]
    panels = [
        ("wind_avoidance_rate", "Wind avoidance",
         "1 - selections / opportunities", (0, 1.15),
         "wind_avoidance_opportunities"),
        ("low_signal_avoidance_rate", "Low-signal avoidance",
         "1 - selections / opportunities", (0, 1.15),
         "low_signal_avoidance_opportunities"),
        ("wind_exposure_steps", "Steps inside wind zones", "steps", None, None),
        ("delivery_rate", "Delivery rate", "fraction of patients",
         (0, 1.05), None),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(15.5, 3.7))
    any_data = False
    for ax, (metric, title, ylabel, ylim, denom) in zip(axes, panels):
        for policy in policies:
            pts = []
            for r in sweep:
                if r["policy"] != policy:
                    continue
                x = density(r)
                y = fnum(r, metric + "_mean")
                if y is None or math.isnan(x):
                    continue
                pts.append((
                    x, y, fnum(r, metric + "_ci95") or 0.0,
                    fnum(r, denom + "_mean") if denom else None,
                ))
            pts.sort(key=lambda t: t[0])
            if not pts:
                continue
            ax.errorbar(
                [q[0] for q in pts], [q[1] for q in pts],
                yerr=[q[2] for q in pts],
                color=POLICY_COLORS[policy], linewidth=2.0, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                capsize=3, elinewidth=1.0,
                label=POLICY_LABELS[policy].replace(chr(10), " "),
            )
            any_data = True
            if denom:
                # learned sits near 1.0 and nearest_path lower, so push their
                # labels apart rather than onto each other's markers
                dy = -14 if policy == "learned" else 12
                for x, y, _, n in pts:
                    if n is not None:
                        ax.annotate(f"{n:.0f}", (x, y),
                                    textcoords="offset points", xytext=(0, dy),
                                    ha="center", fontsize=7.5, color=MUTED)
        style_axis(ax, title, ylabel, ylim)
        ax.set_xlabel("hazard density (x trained)", fontsize=9.5, color=MUTED)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        # Everything right of 1.0x is extrapolation beyond the training range.
        ax.axvspan(1.0, 2.15, color="#f4f4f4", zorder=0)
        if metric.endswith("avoidance_rate"):
            ax.axhline(0.98, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
        # A single legend for the figure: the same two series appear in every
        # panel, and repeating it four times collided with the flat
        # zero-exposure line.
        if metric == "delivery_rate":
            ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle(
        "Hazard density sweep - shaded region is OFF-DISTRIBUTION (training "
        "used 0.5x and 1.0x); small numbers are opportunity counts; 95% CI",
        fontsize=10.5, color=MUTED, y=1.05,
    )
    return save(fig, out_dir, "fig8_hazard_sweep.png") if any_data else None


def fig_latency_sweep(rows, out_dir: Path, zero_delay: Optional[dict] = None
                      ) -> Optional[Path]:
    """Degradation vs command staleness, with the zero-delay reference point.

    The zero-delay anchor comes from the nominal arm of another run (training
    used no delay, so there is no delay_0 scenario), and is passed in rather
    than inferred so the provenance stays explicit.
    """
    sweep = [r for r in rows if r.get("suite") == "latency_sweep"]
    if not sweep:
        return None

    def delay(row) -> float:
        try:
            return float(row["scenario"].split("_")[-1])
        except (ValueError, KeyError, IndexError):
            return float("nan")

    panels = [
        ("delivery_rate", "Delivery rate", "fraction of patients", (0, 1.05)),
        ("triage_efficiency", "Triage efficiency", "acuity-weighted", (0, 1.05)),
        ("drones_landed", "Drones recovered", "count of 5", (0, 5.4)),
        ("wind_avoidance_rate", "Wind avoidance",
         "1 - selections / opportunities", (0, 1.15)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(15.0, 3.7))
    any_data = False
    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        pts = []
        if zero_delay and metric in zero_delay:
            pts.append((0.0, zero_delay[metric][0], zero_delay[metric][1]))
        for r in sweep:
            x = delay(r)
            y = fnum(r, metric + "_mean")
            if y is None or math.isnan(x):
                continue
            pts.append((x, y, fnum(r, metric + "_ci95") or 0.0))
        pts.sort(key=lambda t: t[0])
        if not pts:
            continue
        ax.errorbar(
            [q[0] for q in pts], [q[1] for q in pts], yerr=[q[2] for q in pts],
            color="#2a78d6", linewidth=2.0, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.2, capsize=3,
            elinewidth=1.0,
        )
        any_data = True
        style_axis(ax, title, ylabel, ylim)
        ax.set_xlabel("command delay (steps of stale state)", fontsize=9.5,
                      color=MUTED)
        ax.set_xticks([0, 1, 2, 4])
        # Everything past 0 is off-distribution: training had no delay.
        ax.axvspan(0.001, 4.3, color="#f4f4f4", zorder=0)
        if metric == "wind_avoidance_rate":
            ax.axhline(0.98, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
    fig.suptitle(
        "Command-latency robustness (learned policy) - ALL delays are "
        "OFF-DISTRIBUTION; delay 0 is the pooled nominal reference; 95% CI",
        fontsize=10.5, color=MUTED, y=1.05,
    )
    return save(fig, out_dir, "fig9_latency_sweep.png") if any_data else None


HEADLINE_METRICS = [
    ("delivery_rate", "Delivery rate", "{:.3f}"),
    ("triage_efficiency", "Triage efficiency", "{:.3f}"),
    ("mission_success", "Mission success", "{:.2f}"),
    ("all_landed", "All landed", "{:.2f}"),
    ("rescue_quality", "Rescue quality", "{:.3f}"),
    ("wind_avoidance_rate", "Wind avoidance", "{:.3f}"),
    ("low_signal_avoidance_rate", "Low-signal avoidance", "{:.3f}"),
    ("reserve_violations", "Reserve violations", "{:.0f}"),
    ("drones_depleted", "Drones depleted", "{:.2f}"),
    ("mean_delivered_response_time", "Mean response (steps)", "{:.1f}"),
]


def cell(row: Optional[dict[str, str]], metric: str, fmt: str) -> str:
    if row is None:
        return "--"
    mean = fnum(row, f"{metric}_mean")
    if mean is None:
        return "--"
    ci = fnum(row, f"{metric}_ci95")
    text = fmt.format(mean)
    return f"{text} ± {fmt.format(ci)}" if ci else text


def write_latex_table(rows, path: Path) -> None:
    policies = [
        p for p in POLICY_ORDER
        if summary_row(rows, "baseline_comparison", "nominal", p) is not None
    ]
    if not policies:
        return
    header = " & ".join(
        ["Metric"] + [POLICY_LABELS[p].replace("\n", " ") for p in policies]
    )
    lines = [
        "% Generated by scripts/make_paper_figures.py",
        "\\begin{tabular}{l" + "r" * len(policies) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for metric, label, fmt in HEADLINE_METRICS:
        cells = [
            cell(summary_row(rows, "baseline_comparison", "nominal", p), metric, fmt)
            for p in policies
        ]
        lines.append(" & ".join([label] + cells).replace("±", "$\\pm$") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    run_dir: Path, sitl_rows, offline_rows, episodes, figures, path: Path
) -> None:
    n_by_policy: dict[tuple, int] = {}
    for row in episodes:
        key = (row.get("suite"), row.get("scenario"), row.get("policy"))
        n_by_policy[key] = n_by_policy.get(key, 0) + 1

    out = [
        f"# Analysis — {run_dir.name}",
        "",
        "Generated by `scripts/make_paper_figures.py`. All figures in "
        "`paper_figures/`. Episode-level data in `tables/episodes.csv`; "
        "per-group means with 95% CI in `tables/summary.csv`.",
        "",
        "## Episode counts",
        "",
        "| suite | scenario | policy | episodes |",
        "|---|---|---|---:|",
    ]
    for (suite, scenario, policy), count in sorted(
        n_by_policy.items(), key=lambda kv: str(kv[0])
    ):
        out.append(f"| {suite} | {scenario} | {policy} | {count} |")

    policies = [
        p for p in POLICY_ORDER
        if summary_row(sitl_rows, "baseline_comparison", "nominal", p) is not None
    ]
    out += [
        "",
        "## Nominal comparison (PX4 SITL)",
        "",
        "| Metric | " + " | ".join(POLICY_LABELS[p].replace("\n", " ")
                                   for p in policies) + " |",
        "|---" * (len(policies) + 1) + "|",
    ]
    for metric, label, fmt in HEADLINE_METRICS:
        cells = [
            cell(summary_row(sitl_rows, "baseline_comparison", "nominal", p),
                 metric, fmt)
            for p in policies
        ]
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Notes for writing up",
        "",
        "- **Read every avoidance rate next to its opportunity count.** The env "
        "scores `1 - selections / max(1, opportunities)`, so a policy that never "
        "approached a hazard scores a vacuous 1.0. The opportunity counts are "
        "plotted under each bar in fig2 and are columns in `episodes.csv`.",
        "- **The opportunity denominator is NON-MONOTONE in hazard density.** "
        "The env counts an avoidance opportunity only when a drone's valid "
        "destinations contain BOTH a hazard cell and a hazard-free one, i.e. a "
        "genuine choice. So opportunities rise with density (rarely adjacent to "
        "hazard -> often adjacent with an escape) and then FALL again once the "
        "map saturates and every neighbour is hazardous. Measured: 24 / 36 / 90 "
        "/ 43 opportunities at 0.5x / 1.0x / 1.5x / 2.0x. Do NOT read the "
        "opportunity count as a proxy for hazard pressure, and note that at high "
        "density a drone can be FORCED into a hazard — that shows up in "
        "`wind_exposure_steps`, not in the avoidance rate. Report the rate and "
        "the exposure steps together.",
        "- Hazard entries are near zero for a well-trained policy by design "
        "(stage-2 training gates demand >= 0.98 avoidance), so avoidance rate, "
        "not entry count, is the reportable quantity.",
        "- **`mission_success` is a trap — never quote it alone.** It requires "
        "no UNRESOLVED patients plus all five drones landed, and a patient who "
        "DIES counts as resolved. So a policy that lets patients die quickly "
        "and then lands cleanly scores mission success: in this run "
        "priority_path (delivery 0.17) scored HIGHER mission_success than the "
        "learned policy (delivery 0.70). Always read it beside delivery_rate "
        "and triage_efficiency, and prefer `drones_landed` (continuous) over "
        "`all_landed` for fleet-recovery claims.",
        "- Episode length is bounded by the mission-energy ledger "
        "(100 charge / 0.20 per step = 500 steps of flight), not by the "
        "800-step mission budget; episodes typically end on fleet depletion or "
        "on all drones landing.",
    ]

    if offline_rows:
        learned_offline = summary_row(
            offline_rows, "baseline_comparison", "nominal", "learned")
        learned_sitl = summary_row(
            sitl_rows, "baseline_comparison", "nominal", "learned")
        out += [
            "",
            "## Sim-to-sim transfer (learned policy)",
            "",
            "| Metric | Abstract env (offline) | PX4 SITL |",
            "|---|---|---|",
        ]
        for metric, label, fmt in HEADLINE_METRICS:
            out.append(
                f"| {label} | {cell(learned_offline, metric, fmt)} "
                f"| {cell(learned_sitl, metric, fmt)} |"
            )
        out += [
            "",
            "Same world model and same policy weights on both sides; the only "
            "difference is that SITL executes each grid move as a real PX4 "
            "flight, so tracking jitter and flight time consume ledger energy "
            "and patient timers that abstract stepping does not.",
        ]

    out += ["", "## Figures", ""]
    for name, produced in figures.items():
        out.append(f"- `{name}` — {'written' if produced else 'SKIPPED (no data)'}")
    out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="SITL run directory (contains tables/)")
    parser.add_argument("--offline-dir", type=Path, default=None,
                        help="Offline companion run directory for the "
                             "transfer figure")
    args = parser.parse_args()

    run_dir = args.run_dir
    tables = run_dir / "tables"
    sitl_rows = read_csv(tables / "summary.csv")
    episodes = read_csv(tables / "episodes.csv")
    if not sitl_rows:
        print(f"No summary.csv under {tables} — nothing to plot yet.")
        return 1

    offline_rows = []
    if args.offline_dir:
        offline_rows = read_csv(args.offline_dir / "tables" / "summary.csv")
        if not offline_rows:
            print(f"warning: no offline summary under {args.offline_dir}")

    out_dir = run_dir / "paper_figures"
    figures = {
        "fig1_nominal_comparison.png": fig_nominal(sitl_rows, out_dir),
        "fig2_hazard_discipline.png": fig_hazard(sitl_rows, out_dir),
        "fig3_energy_discipline.png": fig_energy(sitl_rows, out_dir),
        "fig4_triage_by_acuity.png": fig_triage_by_acuity(sitl_rows, out_dir),
        "fig5_transfer_offline_vs_sitl.png": fig_transfer(
            sitl_rows, offline_rows, out_dir),
        "fig6_battery_sweep.png": fig_battery_sweep(sitl_rows, out_dir),
        "fig8_hazard_sweep.png": fig_hazard_sweep(sitl_rows, out_dir),
        "fig9_latency_sweep.png": fig_latency_sweep(
            sitl_rows, out_dir, zero_delay=ZERO_DELAY_REFERENCE),
    }
    try:
        figures["fig7_trajectories.png"] = fig_trajectories(run_dir, out_dir)
    except Exception as exc:  # never let the qualitative extra break the run
        print(f"warning: fig7 trajectories failed: {exc}")
        figures["fig7_trajectories.png"] = None
    write_latex_table(sitl_rows, out_dir / "table_main.tex")
    write_analysis(run_dir, sitl_rows, offline_rows, episodes, figures,
                   run_dir / "analysis.md")

    for name, path in figures.items():
        print(f"{'ok  ' if path else 'skip'} {name}")
    print(f"tables -> {out_dir / 'table_main.tex'}")
    print(f"analysis -> {run_dir / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
