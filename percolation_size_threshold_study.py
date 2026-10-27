
"""
Monte Carlo percolation study for combinatorial thin-film microstructures.

Required project modules:
    morphology.py
    network.py
    area_fraction.py

Outputs:
    results/percolation_size_threshold_study/
        percolation_trial_results.csv
        percolation_summary_by_nominal_fraction.csv
        percolation_summary_by_actual_fraction.csv
        percolation_threshold_estimates.csv
        run_metadata.json
        figures/
            fig1_spanning_probability_vs_nominal_fraction.png/.pdf
            fig2_spanning_probability_vs_actual_area_fraction.png/.pdf
            fig3_threshold_comparison_by_particle_size.png/.pdf
            fig4_nominal_to_actual_area_fraction.png/.pdf
        representative_networks/
            32 representative particle/network plots (.png and .pdf)

IMPORTANT:
- "Nominal fraction" is the sum of individual disk areas divided by domain
  area. It can exceed 1 when overlapping particles are allowed.
- "Actual fraction" is the rasterized occupied 2D area fraction. It is
  NOT automatically a 3D volume fraction.
- A spanning cluster means geometric left-to-right connectivity.
- Check raster resolution, domain size, and fraction-grid sensitivity
  before reporting final scientific conclusions.

Run from the project root:
    python percolation_size_threshold_study.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.optimize import minimize
from scipy.special import expit

from morphology import generate_grains
from network import build_contact_graph, find_spanning_cluster
from area_fraction import actual_fraction_from_morphology
from electrical import four_probe_resistance_multiterminal


# =====================================================================
# Configuration
# =====================================================================

N_TRIALS = 1000

# Nominal sum-of-disk-area/domain-area ratios.
NOMINAL_FRACTIONS = np.round(
    np.arange(0.4, 1.81, 0.2), 3
).tolist()

PARTICLE_SIZE_RANGES = {
    "0.2-0.4": (0.2, 0.4),
    "0.4-0.6": (0.4, 0.6),
    "0.6-0.8": (0.6, 0.8),
    "0.8-1.0": (0.8, 1.0),
    "1.0-1.5": (1.0, 1.5),
}

DOMAIN_WIDTH = 20.0
DOMAIN_HEIGHT = 20.0
CONTACT_TOLERANCE = 0.0
ALLOW_OVERLAP = True

# Rasterization resolution for actual occupied area fraction.
# Test sensitivity at, for example, 25, 50, and 100 pixels/unit.
AREA_PIXELS_PER_UNIT = 50

MASTER_SEED = 20261027

OUTPUT_DIR = Path("results_1000") / "percolation_size_threshold_study"
FIGURE_DIR = OUTPUT_DIR / "figures"
NETWORK_DIR = OUTPUT_DIR / "representative_networks"

N_BOOTSTRAP = 200
CONFIDENCE_LEVEL = 0.95

FIGURE_DPI = 300
SAVE_PDF = True
SAVE_REPRESENTATIVE_NETWORKS = True

# Electrical model parameters. Morphology coordinates are assumed to be µm.
TARGET_VALID_RESISTANCE_MEASUREMENTS = 100
GRAIN_BOUNDARY_RESISTANCE_OHM = 1.0
GRAIN_RESISTIVITY_OHM_M = 1e-5
FILM_THICKNESS_M = 92e-9
EFFECTIVE_WIDTH_M = 200e-9
PROBE_SPACING_MODEL_UNITS = 1.0
PROBE_CONTACT_RADIUS_MODEL_UNITS = 0.0
RESISTANCE_CURRENT_A = 1.0
LENGTH_UNIT_TO_M = 1e-6

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "savefig.dpi": FIGURE_DPI,
})


# =====================================================================
# General helpers
# =====================================================================

def set_plot_style():
    """Set consistent plotting defaults."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_figure(fig, stem: str, folder: Path = FIGURE_DIR):
    """Save a figure as publication-resolution PNG and optional PDF."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        folder / f"{stem}.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    if SAVE_PDF:
        fig.savefig(
            folder / f"{stem}.pdf",
            bbox_inches="tight",
        )
    plt.close(fig)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        return float("nan"), float("nan")

    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = (
        z
        * math.sqrt(
            p * (1 - p) / trials
            + z**2 / (4 * trials**2)
        )
        / denom
    )

    return max(0.0, centre - half), min(1.0, centre + half)


# =====================================================================
# Logistic threshold fitting
# =====================================================================

def logistic_probability(
    x: np.ndarray,
    threshold: float,
    slope: float,
) -> np.ndarray:
    """Probability curve whose P=0.5 point is threshold."""
    return expit(slope * (np.asarray(x, dtype=float) - threshold))


def fit_logistic_threshold(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Fit a monotonic logistic curve using trial-level binary outcomes.

    Returns:
        threshold: predictor at fitted P(span)=0.5
        slope: fitted logistic slope
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 10 or np.unique(y).size < 2:
        raise ValueError("Insufficient outcome variation for logistic fit.")

    xmin = float(x.min())
    xmax = float(x.max())
    scale = xmax - xmin

    if scale <= 0:
        raise ValueError("Predictor has no variation.")

    def objective(params):
        threshold_scaled, log_slope_scaled = params
        threshold = xmin + threshold_scaled * scale
        slope = np.exp(log_slope_scaled) / scale

        p = np.clip(
            logistic_probability(x, threshold, slope),
            1e-10,
            1 - 1e-10,
        )
        return float(
            -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        )

    result = minimize(
        objective,
        x0=np.array([0.5, np.log(8.0)]),
        method="L-BFGS-B",
        bounds=[
            (-2.0, 3.0),
            (np.log(0.05), np.log(1000.0)),
        ],
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Logistic fit failed: {result.message}")

    threshold = xmin + result.x[0] * scale
    slope = np.exp(result.x[1]) / scale

    if threshold < xmin or threshold > xmax:
        raise ValueError(
            "Fitted threshold lies outside the sampled predictor range."
        )

    return float(threshold), float(slope)


def bootstrap_threshold_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> tuple[float, float, int]:
    """Estimate a trial-level bootstrap confidence interval."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    estimates = []
    n = len(x)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        try:
            threshold, _ = fit_logistic_threshold(
                x[indices], y[indices]
            )
            estimates.append(threshold)
        except (ValueError, FloatingPointError):
            continue

    if len(estimates) < max(30, int(0.5 * n_bootstrap)):
        return float("nan"), float("nan"), len(estimates)

    alpha = 1 - confidence_level
    low, high = np.quantile(
        estimates,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(low), float(high), len(estimates)


# =====================================================================
# Representative particle and network plot
# =====================================================================

def plot_representative_network(
    morphology,
    graph,
    spanning_cluster,
    spans: bool,
    size_label: str,
    diameter_range: tuple[float, float],
    nominal_fraction: float,
    actual_fraction: float,
    seed: int,
):
    """
    Save one representative morphology/contact-network figure per
    particle-size range and nominal fraction.

    Left panel: grain circles.
    Right panel: grain centers connected by graph edges.
    The spanning cluster is highlighted when present.
    """
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    safe_label = size_label.replace(".", "p").replace("-", "_")
    fraction_label = f"{nominal_fraction:.1f}".replace(".", "p")

    stem = (
        f"network_size_{safe_label}"
        f"_eta_{fraction_label}"
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.2, 4.5),
        constrained_layout=True,
    )
    ax_particles, ax_network = axes

    centers = np.asarray(morphology.centers)
    radii = np.asarray(morphology.radii)

    # Left panel: circles.
    for center, radius in zip(centers, radii):
        ax_particles.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                float(radius),
                fill=False,
                linewidth=0.65,
                alpha=0.8,
            )
        )

    if len(centers):
        ax_particles.scatter(
            centers[:, 0], centers[:, 1],
            s=5, marker=".", zorder=2,
        )

    ax_particles.set_title("Particle distribution")
    ax_particles.set_xlabel("x (model units)")
    ax_particles.set_ylabel("y (model units)")
    ax_particles.set_xlim(0, morphology.width)
    ax_particles.set_ylim(0, morphology.height)
    ax_particles.set_aspect("equal", adjustable="box")
    ax_particles.grid(alpha=0.15)

    # Right panel: contact graph.
    for i, j in graph.edges():
        i, j = int(i), int(j)
        ax_network.plot(
            [centers[i, 0], centers[j, 0]],
            [centers[i, 1], centers[j, 1]],
            linewidth=0.55,
            alpha=0.6,
            zorder=1,
        )

    if len(centers):
        ax_network.scatter(
            centers[:, 0], centers[:, 1],
            s=8,
            zorder=2,
            linewidths=0.25,
            label="Grains",
        )

    if spans and spanning_cluster:
        ids = np.asarray(sorted(spanning_cluster), dtype=int)
        ax_network.scatter(
            centers[ids, 0],
            centers[ids, 1],
            s=20,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label="Spanning cluster",
        )
        ax_network.legend(frameon=False, loc="best")

    ax_network.set_title("Grain-contact network")
    ax_network.set_xlabel("x (model units)")
    ax_network.set_ylabel("y (model units)")
    ax_network.set_xlim(0, morphology.width)
    ax_network.set_ylim(0, morphology.height)
    ax_network.set_aspect("equal", adjustable="box")
    ax_network.grid(alpha=0.15)

    span_label = "spanning" if spans else "not spanning"

    fig.suptitle(
        f"Particle size {size_label} | "
        f"nominal fraction η={nominal_fraction:.2f} | "
        f"actual area fraction={actual_fraction:.3f} | "
        f"{span_label}",
        fontsize=12,
    )

    fig.savefig(
        NETWORK_DIR / f"{stem}.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    if SAVE_PDF:
        fig.savefig(
            NETWORK_DIR / f"{stem}.pdf",
            bbox_inches="tight",
        )

    plt.close(fig)


# =====================================================================
# Monte Carlo simulation
# =====================================================================

def run_simulations() -> pd.DataFrame:
    """Run all conditions and save the trial-level results."""
    if N_TRIALS < 1:
        raise ValueError("N_TRIALS must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    trial_path = OUTPUT_DIR / "percolation_trial_results.csv"

    if trial_path.exists():
        raise FileExistsError(
            f"{trial_path} already exists.\n"
            "Move/delete the previous results file before starting a "
            "fresh run. This script does not resume an interrupted run."
        )

    n_conditions = len(PARTICLE_SIZE_RANGES) * len(NOMINAL_FRACTIONS)
    total_trials = n_conditions * N_TRIALS

    print(f"Conditions: {n_conditions}")
    print(f"Trials per condition: {N_TRIALS:,}")
    print(f"Total morphologies: {total_trials:,}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("Representative network figures: one per condition")
    print("This may take a long time.\n")

    master_sequence = np.random.SeedSequence(MASTER_SEED)
    condition_sequences = master_sequence.spawn(n_conditions)

    records = []
    condition_index = 0
    start_time = time.time()

    for size_label, diameter_range in PARTICLE_SIZE_RANGES.items():
        for nominal_fraction in NOMINAL_FRACTIONS:
            condition_index += 1
            condition_rng = np.random.default_rng(
                condition_sequences[condition_index - 1]
            )

            seeds = condition_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=N_TRIALS,
                dtype=np.uint32,
            )

            print(
                f"[{condition_index}/{n_conditions}] "
                f"size={size_label}, "
                f"nominal fraction={nominal_fraction:.2f}"
            )

            # Reset the counter for each size/fraction condition.
            resistance_valid_count = 0

            for trial_index, seed in enumerate(seeds, start=1):
                morphology = generate_grains(
                    width=DOMAIN_WIDTH,
                    height=DOMAIN_HEIGHT,
                    target_fraction=float(nominal_fraction),
                    diameter_range=diameter_range,
                    seed=int(seed),
                    allow_overlap=ALLOW_OVERLAP,
                )

                graph = build_contact_graph(
                    morphology,
                    contact_tolerance=CONTACT_TOLERANCE,
                )

                spans, spanning_cluster = find_spanning_cluster(
                    morphology, graph
                )

                actual_fraction = actual_fraction_from_morphology(
                    morphology,
                    pixels_per_unit=AREA_PIXELS_PER_UNIT,
                )

                # Save the first morphology only for each condition.
                if (
                    SAVE_REPRESENTATIVE_NETWORKS
                    and trial_index == 1
                ):
                    plot_representative_network(
                        morphology=morphology,
                        graph=graph,
                        spanning_cluster=spanning_cluster,
                        spans=bool(spans),
                        size_label=size_label,
                        diameter_range=diameter_range,
                        nominal_fraction=float(nominal_fraction),
                        actual_fraction=float(actual_fraction),
                        seed=int(seed),
                    )

                resistance_value = np.nan
                resistance_status = "not_attempted"
                resistance_reason = ""
                #resistance_valid_count = 0

                # Only calculate electrical resistance for geometrically
                # spanning morphologies, and stop after 100 valid positive
                # four-probe readings for this condition.
                if bool(spans) and (
                    resistance_valid_count < TARGET_VALID_RESISTANCE_MEASUREMENTS
                ):
                    resistance_result = four_probe_resistance_multiterminal(
                        morphology=morphology,
                        graph=graph,
                        probe_spacing=PROBE_SPACING_MODEL_UNITS,
                        current_amp=RESISTANCE_CURRENT_A,
                        grain_boundary_resistance_ohm=GRAIN_BOUNDARY_RESISTANCE_OHM,
                        grain_resistivity_ohm_m=GRAIN_RESISTIVITY_OHM_M,
                        film_thickness_m=FILM_THICKNESS_M,
                        effective_width_m=EFFECTIVE_WIDTH_M,
                        probe_contact_radius=PROBE_CONTACT_RADIUS_MODEL_UNITS,
                        length_unit_to_m=LENGTH_UNIT_TO_M,
                    )
                    candidate = float(
                        resistance_result.get("resistance_ohm", np.nan)
                    )
                    resistance_reason = str(
                        resistance_result.get("reason") or ""
                    )

                    if (
                        resistance_result.get("percolates", False)
                        and np.isfinite(candidate)
                        and candidate > 0
                    ):
                        resistance_value = candidate
                        resistance_status = "valid_positive"
                        resistance_valid_count += 1
                    elif np.isfinite(candidate) and candidate < 0:
                        resistance_status = "rejected_negative"
                    else:
                        resistance_status = "invalid_electrical_network"

                records.append({
                    "particle_size_range": size_label,
                    "diameter_min": diameter_range[0],
                    "diameter_max": diameter_range[1],
                    "nominal_target_fraction": float(nominal_fraction),
                    "nominal_achieved_fraction": float(nominal_fraction),
                    "actual_occupied_area_fraction": float(actual_fraction),
                    "trial": int(trial_index),
                    "seed": int(seed),
                    "grain_count": len(morphology.radii),
                    "contact_edges": int(graph.number_of_edges()),
                    "spans_left_to_right": int(bool(spans)),
                    "spanning_cluster_size": len(spanning_cluster),
                    "resistance_ohm": resistance_value,
                    "resistance_status": resistance_status,
                    "resistance_reason": resistance_reason,
                })

                # Checkpoint every 250 trials within a condition.
                if trial_index % 250 == 0:
                    pd.DataFrame(records).to_csv(
                        trial_path, index=False
                    )
                    elapsed = (time.time() - start_time) / 60
                    print(
                        f"    trial {trial_index}/{N_TRIALS}; "
                        f"elapsed {elapsed:.1f} min"
                    )

            print(
                f"  Valid positive resistance readings: "
                f"{resistance_valid_count}/{TARGET_VALID_RESISTANCE_MEASUREMENTS}"
            )

            # Checkpoint after each condition.
            pd.DataFrame(records).to_csv(
                trial_path, index=False
            )

    results = pd.DataFrame(records)
    results.to_csv(trial_path, index=False)

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nSimulation complete in {elapsed_hours:.2f} hours.")
    print(f"Trial results saved to {trial_path.resolve()}")

    return results


# =====================================================================
# Summary tables
# =====================================================================

def summarize_nominal_fraction(trials: pd.DataFrame) -> pd.DataFrame:
    """Spanning probability and Wilson confidence intervals by nominal fraction."""
    rows = []

    keys = ["particle_size_range", "nominal_target_fraction"]

    for (size_label, fraction), group in trials.groupby(keys, sort=True):
        n = len(group)
        successes = int(group["spans_left_to_right"].sum())
        p = successes / n
        low, high = wilson_interval(successes, n)

        rows.append({
            "particle_size_range": size_label,
            "nominal_target_fraction": fraction,
            "n_trials": n,
            "n_spanning": successes,
            "spanning_probability": p,
            "spanning_ci_lower": low,
            "spanning_ci_upper": high,
            "mean_actual_area_fraction":
                group["actual_occupied_area_fraction"].mean(),
            "std_actual_area_fraction":
                group["actual_occupied_area_fraction"].std(),
            "mean_grain_count": group["grain_count"].mean(),
            "mean_contact_edges": group["contact_edges"].mean(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "percolation_summary_by_nominal_fraction.csv",
        index=False,
    )
    return result


def summarize_actual_fraction(
    trials: pd.DataFrame,
    bin_width: float = 0.025,
) -> pd.DataFrame:
    """
    Bin trials by actual occupied area fraction and calculate
    spanning probability within each bin.
    """
    df = trials.copy()
    values = df["actual_occupied_area_fraction"].to_numpy()

    lower = np.floor(values / bin_width) * bin_width
    df["actual_fraction_bin_lower"] = lower
    df["actual_fraction_bin_centre"] = lower + bin_width / 2

    rows = []
    keys = ["particle_size_range", "actual_fraction_bin_lower"]

    for (size_label, bin_lower), group in df.groupby(keys, sort=True):
        n = len(group)

        # Avoid noisy bins with very few observations.
        if n < 20:
            continue

        successes = int(group["spans_left_to_right"].sum())
        p = successes / n
        low, high = wilson_interval(successes, n)

        rows.append({
            "particle_size_range": size_label,
            "actual_fraction_bin_lower": bin_lower,
            "actual_fraction_bin_centre": bin_lower + bin_width / 2,
            "n_trials": n,
            "n_spanning": successes,
            "spanning_probability": p,
            "spanning_ci_lower": low,
            "spanning_ci_upper": high,
            "mean_actual_fraction":
                group["actual_occupied_area_fraction"].mean(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "percolation_summary_by_actual_fraction.csv",
        index=False,
    )
    return result


def estimate_thresholds(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate P(span)=0.5 threshold against both nominal and actual
    occupied area fraction, with trial-level bootstrap intervals.
    """
    rows = []
    rng = np.random.default_rng(MASTER_SEED + 1)

    axes = {
        "nominal_disk_fraction": "nominal_target_fraction",
        "actual_occupied_area_fraction": "actual_occupied_area_fraction",
    }

    for size_label in PARTICLE_SIZE_RANGES:
        subset = trials[
            trials["particle_size_range"] == size_label
        ]

        y = subset["spans_left_to_right"].to_numpy(dtype=int)

        for axis_name, x_column in axes.items():
            x = subset[x_column].to_numpy(dtype=float)

            base = {
                "particle_size_range": size_label,
                "threshold_axis": axis_name,
                "n_trials": len(subset),
                "n_spanning": int(y.sum()),
            }

            try:
                threshold, slope = fit_logistic_threshold(x, y)
                ci_low, ci_high, n_valid = bootstrap_threshold_interval(
                    x,
                    y,
                    n_bootstrap=N_BOOTSTRAP,
                    rng=rng,
                    confidence_level=CONFIDENCE_LEVEL,
                )

                rows.append({
                    **base,
                    "threshold_p_span_0.5": threshold,
                    "logistic_slope": slope,
                    "bootstrap_ci_lower": ci_low,
                    "bootstrap_ci_upper": ci_high,
                    "bootstrap_valid_fits": n_valid,
                    "fit_status": "ok",
                })

            except (ValueError, FloatingPointError) as exc:
                rows.append({
                    **base,
                    "threshold_p_span_0.5": np.nan,
                    "logistic_slope": np.nan,
                    "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan,
                    "bootstrap_valid_fits": 0,
                    "fit_status": f"failed: {exc}",
                })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "percolation_threshold_estimates.csv",
        index=False,
    )
    return result


# =====================================================================
# Electrical resistance summaries and plots
# =====================================================================

def summarize_resistance(trials: pd.DataFrame) -> pd.DataFrame:
    """Summarize accepted positive four-probe resistance measurements."""
    valid = trials[
        (trials["resistance_status"] == "valid_positive")
        & np.isfinite(trials["resistance_ohm"])
        & (trials["resistance_ohm"] > 0)
    ].copy()

    rows = []
    keys = ["particle_size_range", "nominal_target_fraction"]

    for (size_label, fraction), group in trials.groupby(keys, sort=True):
        accepted = valid[
            (valid["particle_size_range"] == size_label)
            & (valid["nominal_target_fraction"] == fraction)
        ]
        values = accepted["resistance_ohm"].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]

        attempted = group["resistance_status"].isin(
            ["valid_positive", "rejected_negative", "invalid_electrical_network"]
        ).sum()
        rejected_negative = (
            group["resistance_status"] == "rejected_negative"
        ).sum()
        invalid = (
            group["resistance_status"] == "invalid_electrical_network"
        ).sum()

        rows.append({
            "particle_size_range": size_label,
            "nominal_target_fraction": fraction,
            "n_valid_positive_resistance": int(len(values)),
            "n_resistance_attempts": int(attempted),
            "n_negative_rejected": int(rejected_negative),
            "n_invalid_electrical_network": int(invalid),
            "mean_resistance_ohm": float(np.mean(values)) if len(values) else np.nan,
            "median_resistance_ohm": float(np.median(values)) if len(values) else np.nan,
            "std_resistance_ohm": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "q25_resistance_ohm": float(np.quantile(values, 0.25)) if len(values) else np.nan,
            "q75_resistance_ohm": float(np.quantile(values, 0.75)) if len(values) else np.nan,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        OUTPUT_DIR / "resistance_summary_by_condition.csv",
        index=False,
    )

    valid.to_csv(
        OUTPUT_DIR / "valid_resistance_measurements.csv",
        index=False,
    )
    return summary


def plot_resistance_vs_nominal(summary: pd.DataFrame):
    """Plot median resistance and interquartile range versus nominal fraction."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in PARTICLE_SIZE_RANGES:
        group = summary[
            (summary["particle_size_range"] == label)
            & (summary["n_valid_positive_resistance"] > 0)
        ].sort_values("nominal_target_fraction")

        if group.empty:
            continue

        x = group["nominal_target_fraction"].to_numpy(dtype=float)
        median = group["median_resistance_ohm"].to_numpy(dtype=float)
        q25 = group["q25_resistance_ohm"].to_numpy(dtype=float)
        q75 = group["q75_resistance_ohm"].to_numpy(dtype=float)

        ax.plot(x, median, marker="o", markersize=4, label=label)
        ax.fill_between(x, q25, q75, alpha=0.16)

    ax.set_xlabel("Nominal disk fraction, η")
    ax.set_ylabel("Four-probe transfer resistance (Ω)")
    ax.set_title("Positive four-probe resistance versus nominal fraction")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig5_resistance_vs_nominal_fraction")


def plot_resistance_vs_actual(trials: pd.DataFrame):
    """Plot positive resistance against actual occupied area fraction."""
    valid = trials[
        (trials["resistance_status"] == "valid_positive")
        & np.isfinite(trials["resistance_ohm"])
        & (trials["resistance_ohm"] > 0)
    ].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in PARTICLE_SIZE_RANGES:
        group = valid[valid["particle_size_range"] == label]
        if group.empty:
            continue

        # Aggregate by nominal condition to avoid a noisy cloud and make
        # each point represent the mean actual fraction for that condition.
        agg = group.groupby("nominal_target_fraction").agg(
            actual_fraction=("actual_occupied_area_fraction", "mean"),
            median_resistance=("resistance_ohm", "median"),
            q25=("resistance_ohm", lambda s: s.quantile(0.25)),
            q75=("resistance_ohm", lambda s: s.quantile(0.75)),
        ).reset_index().sort_values("actual_fraction")

        ax.plot(
            agg["actual_fraction"],
            agg["median_resistance"],
            marker="o",
            markersize=4,
            label=label,
        )
        ax.fill_between(
            agg["actual_fraction"].to_numpy(dtype=float),
            agg["q25"].to_numpy(dtype=float),
            agg["q75"].to_numpy(dtype=float),
            alpha=0.16,
        )

    ax.set_xlabel("Actual occupied 2D area fraction")
    ax.set_ylabel("Four-probe transfer resistance (Ω)")
    ax.set_title("Positive four-probe resistance versus actual area fraction")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig6_resistance_vs_actual_area_fraction")


def plot_resistance_distributions(trials: pd.DataFrame):
    """Save a four-panel distribution plot, one panel per particle-size range."""
    valid = trials[
        (trials["resistance_status"] == "valid_positive")
        & np.isfinite(trials["resistance_ohm"])
        & (trials["resistance_ohm"] > 0)
    ].copy()

    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 7.0), constrained_layout=True
    )

    for ax, label in zip(axes.flat, PARTICLE_SIZE_RANGES):
        subset = valid[valid["particle_size_range"] == label]

        if subset.empty:
            ax.text(
                0.5, 0.5, "No valid positive readings",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(f"Particle size {label}")
            ax.set_xlabel("Nominal disk fraction, η")
            ax.set_ylabel("Resistance (Ω)")
            continue

        fractions = sorted(subset["nominal_target_fraction"].unique())
        data = [
            subset.loc[
                subset["nominal_target_fraction"] == f,
                "resistance_ohm",
            ].to_numpy(dtype=float)
            for f in fractions
        ]

        ax.boxplot(
            data,
            positions=np.arange(len(fractions)),
            widths=0.62,
            showfliers=False,
            medianprops={"linewidth": 1.4},
        )
        ax.set_xticks(np.arange(len(fractions)))
        ax.set_xticklabels([f"{f:.1f}" for f in fractions])
        ax.set_title(f"Particle size {label}")
        ax.set_xlabel("Nominal disk fraction, η")
        ax.set_ylabel("Resistance (Ω)")
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Distribution of accepted positive four-probe readings")
    save_figure(fig, "fig7_resistance_distributions")


# =====================================================================
# Publication plots
# =====================================================================

def plot_spanning_vs_nominal(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in PARTICLE_SIZE_RANGES:
        group = summary[
            summary["particle_size_range"] == label
        ].sort_values("nominal_target_fraction")

        if group.empty:
            continue

        x = group["nominal_target_fraction"].to_numpy()
        p = group["spanning_probability"].to_numpy()
        low = group["spanning_ci_lower"].to_numpy()
        high = group["spanning_ci_upper"].to_numpy()

        ax.plot(x, p, marker="o", markersize=4, label=label)
        ax.fill_between(x, low, high, alpha=0.16)

    ax.axhline(
        0.5, color="black", linestyle="--", linewidth=1,
        label="P(span) = 0.5",
    )
    ax.set_xlabel("Nominal disk fraction, η")
    ax.set_ylabel("Left-to-right spanning probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Geometric percolation versus nominal disk fraction")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig1_spanning_probability_vs_nominal_fraction")


def plot_spanning_vs_actual(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in PARTICLE_SIZE_RANGES:
        group = summary[
            summary["particle_size_range"] == label
        ].sort_values("mean_actual_fraction")

        if group.empty:
            continue

        x = group["mean_actual_fraction"].to_numpy()
        p = group["spanning_probability"].to_numpy()
        low = group["spanning_ci_lower"].to_numpy()
        high = group["spanning_ci_upper"].to_numpy()

        ax.plot(x, p, marker="o", markersize=4, label=label)
        ax.fill_between(x, low, high, alpha=0.16)

    ax.axhline(
        0.5, color="black", linestyle="--", linewidth=1,
        label="P(span) = 0.5",
    )
    ax.set_xlabel("Actual occupied 2D area fraction")
    ax.set_ylabel("Left-to-right spanning probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Geometric percolation versus actual occupied area")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig2_spanning_probability_vs_actual_area_fraction")


def plot_threshold_comparison(thresholds: pd.DataFrame):
    successful = thresholds[
        thresholds["fit_status"] == "ok"
    ].copy()

    if successful.empty:
        print("No successful threshold fits; skipping threshold plot.")
        return

    axes_info = [
        (
            "nominal_disk_fraction",
            "Nominal disk fraction, η",
        ),
        (
            "actual_occupied_area_fraction",
            "Actual occupied 2D area fraction",
        ),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.5),
        constrained_layout=True,
    )

    labels = list(PARTICLE_SIZE_RANGES.keys())
    positions = np.arange(len(labels))

    for ax, (axis_name, axis_title) in zip(axes, axes_info):
        sub = successful[
            successful["threshold_axis"] == axis_name
        ]

        values = []
        lows = []
        highs = []

        for label in labels:
            row = sub[sub["particle_size_range"] == label]

            if row.empty:
                values.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue

            values.append(float(row.iloc[0]["threshold_p_span_0.5"]))
            lows.append(float(row.iloc[0]["bootstrap_ci_lower"]))
            highs.append(float(row.iloc[0]["bootstrap_ci_upper"]))

        values = np.asarray(values)
        lows = np.asarray(lows)
        highs = np.asarray(highs)

        valid = np.isfinite(values)
        valid_ci = valid & np.isfinite(lows) & np.isfinite(highs)

        # Plot all fitted threshold points.
        ax.plot(
            positions[valid],
            values[valid],
            "o",
            markersize=5,
        )

        # Draw error bars only where bootstrap intervals are available.
        for idx in np.where(valid_ci)[0]:
            lower_error = max(0.0, values[idx] - lows[idx])
            upper_error = max(0.0, highs[idx] - values[idx])
            ax.errorbar(
                positions[idx],
                values[idx],
                yerr=[[lower_error], [upper_error]],
                fmt="none",
                capsize=4,
                linewidth=1.1,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Particle diameter range (model units)")
        ax.set_ylabel(axis_title)
        ax.set_title(axis_title)
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Estimated threshold at P(span) = 0.5")
    save_figure(fig, "fig3_threshold_comparison_by_particle_size")


def plot_actual_fraction_mapping(trials: pd.DataFrame):
    summary = (
        trials.groupby(
            ["particle_size_range", "nominal_target_fraction"],
            sort=True,
        )
        .agg(
            mean_actual=(
                "actual_occupied_area_fraction", "mean"
            ),
            std_actual=(
                "actual_occupied_area_fraction", "std"
            ),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in PARTICLE_SIZE_RANGES:
        group = summary[
            summary["particle_size_range"] == label
        ].sort_values("nominal_target_fraction")

        if group.empty:
            continue

        ax.errorbar(
            group["nominal_target_fraction"],
            group["mean_actual"],
            yerr=group["std_actual"],
            marker="o",
            markersize=4,
            capsize=2,
            label=label,
        )

    ax.set_xlabel("Nominal disk fraction, η")
    ax.set_ylabel("Actual occupied 2D area fraction")
    ax.set_title("Nominal fraction versus actual occupied area")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig4_nominal_to_actual_area_fraction")


# =====================================================================
# Main
# =====================================================================

def main():
    set_plot_style()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_REPRESENTATIVE_NETWORKS:
        NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "trials_per_condition": N_TRIALS,
        "nominal_fractions": NOMINAL_FRACTIONS,
        "particle_size_ranges": PARTICLE_SIZE_RANGES,
        "domain_width": DOMAIN_WIDTH,
        "domain_height": DOMAIN_HEIGHT,
        "contact_tolerance": CONTACT_TOLERANCE,
        "allow_overlap": ALLOW_OVERLAP,
        "area_pixels_per_unit": AREA_PIXELS_PER_UNIT,
        "master_seed": MASTER_SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "confidence_level": CONFIDENCE_LEVEL,
        "representative_network_plots": SAVE_REPRESENTATIVE_NETWORKS,
        "representative_networks_per_condition": 1,
        "threshold_definition": "Predictor at fitted P(span)=0.5",
        "actual_fraction_definition": (
            "Rasterized occupied 2D area fraction; not a 3D volume "
            "fraction without additional morphology assumptions."
        ),
        "electrical_model": {
            "solver": "four_probe_resistance_multiterminal",
            "target_valid_positive_measurements_per_condition": TARGET_VALID_RESISTANCE_MEASUREMENTS,
            "grain_boundary_resistance_ohm": GRAIN_BOUNDARY_RESISTANCE_OHM,
            "grain_resistivity_ohm_m": GRAIN_RESISTIVITY_OHM_M,
            "film_thickness_m": FILM_THICKNESS_M,
            "effective_width_m": EFFECTIVE_WIDTH_M,
            "probe_spacing_model_units": PROBE_SPACING_MODEL_UNITS,
            "probe_contact_radius_model_units": PROBE_CONTACT_RADIUS_MODEL_UNITS,
            "current_amp": RESISTANCE_CURRENT_A,
            "length_unit_to_m": LENGTH_UNIT_TO_M,
            "negative_transfer_resistances": "excluded, never converted with abs()",
        },
    }

    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    trials = run_simulations()

    print("\nCalculating summary tables...")
    nominal_summary = summarize_nominal_fraction(trials)
    actual_summary = summarize_actual_fraction(trials)
    thresholds = estimate_thresholds(trials)
    resistance_summary = summarize_resistance(trials)

    print("Generating publication plots...")
    plot_spanning_vs_nominal(nominal_summary)
    plot_spanning_vs_actual(actual_summary)
    plot_threshold_comparison(thresholds)
    plot_actual_fraction_mapping(trials)
    plot_resistance_vs_nominal(resistance_summary)
    plot_resistance_vs_actual(trials)
    plot_resistance_distributions(trials)

    print("\nThreshold estimates:")
    print(thresholds.to_string(index=False))

    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")
    print(f"Summary plots: {FIGURE_DIR.resolve()}")
    print(
        "Resistance summary: "
        f"{(OUTPUT_DIR / 'resistance_summary_by_condition.csv').resolve()}"
    )
    print(
        "Valid resistance measurements: "
        f"{(OUTPUT_DIR / 'valid_resistance_measurements.csv').resolve()}"
    )

    if SAVE_REPRESENTATIVE_NETWORKS:
        print(f"Representative networks: {NETWORK_DIR.resolve()}")
        print(
            "Expected representative conditions: "
            f"{len(PARTICLE_SIZE_RANGES) * len(NOMINAL_FRACTIONS)}"
        )

    print(
        "\nBefore publication, test sensitivity to domain size, "
        "area-raster resolution, and nominal-fraction spacing."
    )


if __name__ == "__main__":
    main()
