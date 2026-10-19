
"""Monte Carlo simulation for 2D CuS-grain percolation.

For every generated morphology:
1. Generate grains at a requested nominal disk fraction.
2. Estimate actual occupied area fraction using a rasterized union of disks.
3. Construct the grain-contact graph and test left-to-right spanning.
4. Optionally calculate four-probe resistance.
5. Save per-trial records and summary statistics.

Outputs:
- percolation_trial_results.csv
- percolation_summary.csv
- percolation_actual_fraction_bins.csv
- monte_carlo_results.json
- spanning_vs_nominal_fraction.png
- spanning_vs_actual_fraction.png
- resistance_vs_nominal_fraction.png (if electrical calculations are enabled)
- resistance_vs_actual_fraction.png (if electrical calculations are enabled)

The actual fraction is the rasterized occupied AREA fraction of the 2D
simulation domain. It represents a volume fraction only under additional
film-geometry assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morphology import generate_grains
from network import build_contact_graph, find_spanning_cluster
from area_fraction import actual_fraction_from_morphology
from electrical import four_probe_resistance_multiterminal


@dataclass
class MonteCarloPoint:
    """Summary of trials at one nominal disk fraction."""

    nominal_fraction: float
    trials: int

    mean_actual_fraction: float
    std_actual_fraction: float

    spanning_count: int
    spanning_probability: float
    spanning_probability_se: float

    valid_resistance_count: int
    resistance_valid_fraction: float
    negative_resistance_count: int

    resistance_median_ohm: float | None
    resistance_mean_ohm: float | None
    resistance_std_ohm: float | None
    median_positive_resistance_ohm: float | None

    resistance_values_ohm: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binomial_se(successes: int, count: int) -> float:
    """Standard error of an estimated binomial probability."""
    if count <= 0:
        return float("nan")

    p = successes / count
    return float(np.sqrt(p * (1.0 - p) / count))


def _safe_float(value: Any) -> float | None:
    """Convert finite numerical values to float; otherwise return None."""
    if value is None:
        return None

    result = float(value)
    return result if np.isfinite(result) else None


def run_monte_carlo_sweep(
    nominal_fractions: list[float] | np.ndarray,
    *,
    trials_per_fraction: int = 30,
    width: float = 30.0,
    height: float = 20.0,
    diameter_range: tuple[float, float] = (0.2, 0.6),
    contact_tolerance: float = 0.02,
    seed: int = 12345,
    allow_overlap: bool = True,
    probe_spacing: float | None = None,
    current_amp: float = 1.0,
    grain_boundary_resistance_ohm: float = 100.0,
    probe_contact_radius: float = 0.0,
    progress: bool = True,
    grain_resistivity_ohm_m: float = 1e-4,
    film_thickness_m: float = 92e-9,
    effective_width_m: float = 200e-9,
    area_pixels_per_unit: float = 50,
    output_dir: str | Path = ".",
) -> tuple[list[MonteCarloPoint], pd.DataFrame]:
    """Run Monte Carlo trials and save per-trial results.

    All geometric lengths must use the same unit.

    Parameters
    ----------
    nominal_fractions:
        Target sum(pi*r**2)/(width*height). May exceed 1 when disks overlap.

    trials_per_fraction:
        Number of independent morphology realizations per nominal fraction.

    width, height:
        Simulation-domain dimensions.

    diameter_range:
        Minimum and maximum grain diameters, in the morphology length unit.

    contact_tolerance:
        Allowed additional gap for grain contacts.

    seed:
        Master seed for reproducible independent trials.

    probe_spacing:
        If None, only geometric percolation is calculated.
        Otherwise, also calculate four-probe resistance.

    area_pixels_per_unit:
        Raster resolution used to estimate the occupied fraction.

    output_dir:
        Directory for CSV and JSON output.

    Returns
    -------
    results:
        Summary records, one per nominal fraction.

    trial_df:
        DataFrame with one row per generated morphology.
    """
    fractions = np.asarray(nominal_fractions, dtype=float)

    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("nominal_fractions must be a non-empty 1D sequence.")

    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("Nominal fractions must be finite and non-negative.")

    if not allow_overlap and np.any(fractions > 1):
        raise ValueError("Fractions above 1 require allow_overlap=True.")

    if trials_per_fraction < 1:
        raise ValueError("trials_per_fraction must be at least 1.")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if contact_tolerance < 0:
        raise ValueError("contact_tolerance must be non-negative.")

    if area_pixels_per_unit <= 0:
        raise ValueError("area_pixels_per_unit must be positive.")

    if probe_spacing is not None:
        if probe_spacing <= 0 or width < 3 * probe_spacing:
            raise ValueError(
                "probe_spacing must be positive and fit three probe "
                "intervals inside the domain."
            )

    if len(diameter_range) != 2 or not (
        0 < diameter_range[0] <= diameter_range[1]
    ):
        raise ValueError("diameter_range must satisfy 0 < min <= max.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create independent reproducible seeds for all trials.
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(
        len(fractions) * trials_per_fraction
    )

    trial_records: list[dict[str, Any]] = []
    results: list[MonteCarloPoint] = []

    child_index = 0

    for fraction_index, requested_fraction in enumerate(fractions):
        spanning_flags: list[bool] = []
        actual_fractions: list[float] = []
        resistance_values: list[float] = []
        negative_resistance_count = 0

        for trial_index in range(trials_per_fraction):
            trial_seed = int(
                child_sequences[child_index].generate_state(1)[0]
            )
            child_index += 1

            microstructure = generate_grains(
                width=width,
                height=height,
                target_fraction=float(requested_fraction),
                diameter_range=diameter_range,
                seed=trial_seed,
                allow_overlap=allow_overlap,
            )

            # Calculate the actual union-area fraction for this morphology.
            actual_fraction = actual_fraction_from_morphology(
                microstructure,
                pixels_per_unit=area_pixels_per_unit,
            )

            graph = build_contact_graph(
                microstructure,
                contact_tolerance=contact_tolerance,
            )

            spans, _ = find_spanning_cluster(
                microstructure,
                graph,
            )
            spans = bool(spans)

            actual_fractions.append(actual_fraction)
            spanning_flags.append(spans)

            resistance = float("nan")
            electrical_percolates: bool | None = None
            electrical_error: str | None = None

            if probe_spacing is not None:
                try:
                    electrical = four_probe_resistance_multiterminal(
                        morphology=microstructure,
                        graph=graph,
                        probe_spacing=probe_spacing,
                        current_amp=current_amp,
                        grain_boundary_resistance_ohm=(
                            grain_boundary_resistance_ohm
                        ),
                        grain_resistivity_ohm_m=grain_resistivity_ohm_m,
                        film_thickness_m=film_thickness_m,
                        effective_width_m=effective_width_m,
                        probe_contact_radius=probe_contact_radius,
                    )

                    resistance = float(electrical["resistance_ohm"])
                    electrical_percolates = bool(
                        electrical.get("percolates", False)
                    )

                    # Preserve the signed resistance for diagnostics.
                    if (
                        electrical_percolates
                        and np.isfinite(resistance)
                    ):
                        resistance_values.append(resistance)

                        if resistance < 0:
                            negative_resistance_count += 1

                except (ValueError, RuntimeError, KeyError) as exc:
                    electrical_error = str(exc)

            # Save one row for every morphology, including failed electrical
            # calculations, so trial accounting remains transparent.
            trial_records.append(
                {
                    "nominal_fraction": float(requested_fraction),
                    "actual_fraction": float(actual_fraction),
                    "grain_count": len(microstructure.centers),
                    "spans": spans,
                    "trial": trial_index,
                    "fraction_index": fraction_index,
                    "seed": trial_seed,
                    "resistance_ohm": _safe_float(resistance),
                    "electrical_percolates": electrical_percolates,
                    "electrical_error": electrical_error,
                }
            )

        spanning_count = int(np.count_nonzero(spanning_flags))
        probability = spanning_count / trials_per_fraction
        probability_se = _binomial_se(
            spanning_count,
            trials_per_fraction,
        )

        mean_actual = float(np.mean(actual_fractions))
        std_actual = (
            float(np.std(actual_fractions, ddof=1))
            if len(actual_fractions) > 1
            else 0.0
        )

        if resistance_values:
            resistance_median = float(np.median(resistance_values))
            resistance_mean = float(np.mean(resistance_values))
            resistance_std = (
                float(np.std(resistance_values, ddof=1))
                if len(resistance_values) > 1
                else 0.0
            )

            positive_values = [
                value for value in resistance_values if value > 0
            ]
            positive_median = (
                float(np.median(positive_values))
                if positive_values
                else None
            )
        else:
            resistance_median = None
            resistance_mean = None
            resistance_std = None
            positive_median = None

        point = MonteCarloPoint(
            nominal_fraction=float(requested_fraction),
            trials=trials_per_fraction,
            mean_actual_fraction=mean_actual,
            std_actual_fraction=std_actual,
            spanning_count=spanning_count,
            spanning_probability=float(probability),
            spanning_probability_se=probability_se,
            valid_resistance_count=len(resistance_values),
            resistance_valid_fraction=(
                len(resistance_values) / trials_per_fraction
            ),
            negative_resistance_count=negative_resistance_count,
            resistance_median_ohm=resistance_median,
            resistance_mean_ohm=resistance_mean,
            resistance_std_ohm=resistance_std,
            median_positive_resistance_ohm=positive_median,
            resistance_values_ohm=resistance_values,
        )

        results.append(point)

        if progress:
            print(
                f"nominal={requested_fraction:.3f} | "
                f"actual={mean_actual:.4f} +/- {std_actual:.4f} | "
                f"spanning={spanning_count}/{trials_per_fraction} "
                f"(p={probability:.3f} +/- {probability_se:.3f}) | "
                f"valid R={len(resistance_values)}/{trials_per_fraction} | "
                f"negative R={negative_resistance_count}"
            )

    # Save once, after all fractions have been processed.
    trial_df = pd.DataFrame(trial_records)
    trial_df.to_csv(
        output_path / "percolation_trial_results.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        [result.to_dict() for result in results]
    )
    summary_df.to_csv(
        output_path / "percolation_summary.csv",
        index=False,
    )

    with open(
        output_path / "monte_carlo_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [result.to_dict() for result in results],
            file,
            indent=2,
        )

    if progress:
        print(f"\nSaved per-trial data to: {output_path.resolve()}")
        print(f"Saved summary data to: {(output_path / 'percolation_summary.csv').resolve()}")

    return results, trial_df


def summarize_actual_fraction_bins(
    trial_df: pd.DataFrame,
    *,
    n_bins: int = 15,
) -> pd.DataFrame:
    """Estimate spanning probability in bins of actual occupied fraction.

    Empty bins are omitted. The returned standard error is a binomial
    standard error based on the trials in each bin.
    """
    required = {"actual_fraction", "spans"}
    missing = required - set(trial_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    df = trial_df.dropna(subset=["actual_fraction", "spans"]).copy()

    if df.empty:
        raise ValueError("No valid trial rows to summarize.")

    if df["actual_fraction"].nunique() < 2:
        # All actual fractions are identical; use one group.
        df["fraction_bin"] = "all"
        summary = (
            df.groupby("fraction_bin", observed=True)
            .agg(
                mean_actual_fraction=("actual_fraction", "mean"),
                trials=("spans", "size"),
                spanning_count=("spans", "sum"),
                spanning_probability=("spans", "mean"),
            )
            .reset_index()
        )
    else:
        df["fraction_bin"] = pd.cut(
            df["actual_fraction"],
            bins=n_bins,
            include_lowest=True,
            duplicates="drop",
        )

        summary = (
            df.groupby("fraction_bin", observed=True)
            .agg(
                mean_actual_fraction=("actual_fraction", "mean"),
                trials=("spans", "size"),
                spanning_count=("spans", "sum"),
                spanning_probability=("spans", "mean"),
            )
            .reset_index()
        )

    summary["spanning_probability_se"] = [
        _binomial_se(int(k), int(n))
        for k, n in zip(
            summary["spanning_count"],
            summary["trials"],
        )
    ]

    return summary


def plot_monte_carlo_results(
    results: list[MonteCarloPoint],
    trial_df: pd.DataFrame,
    *,
    output_dir: str | Path = ".",
    actual_fraction_bins: int = 15,
):
    """Save plots versus nominal and actual occupied fraction."""
    if not results:
        raise ValueError("No Monte Carlo results to plot.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [result.to_dict() for result in results]
    )

    # Plot 1: spanning probability against nominal fraction.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        summary_df["nominal_fraction"],
        summary_df["spanning_probability"],
        yerr=summary_df["spanning_probability_se"],
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("Nominal disk fraction, η")
    ax.set_ylabel("Left-to-right spanning probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Geometric percolation vs nominal fraction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_path / "spanning_vs_nominal_fraction.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

    # Plot 2: spanning probability against actual fraction, using
    # the actual fraction measured for every individual morphology.
    actual_summary = summarize_actual_fraction_bins(
        trial_df,
        n_bins=actual_fraction_bins,
    )
    actual_summary.to_csv(
        output_path / "percolation_actual_fraction_bins.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        actual_summary["mean_actual_fraction"],
        actual_summary["spanning_probability"],
        yerr=actual_summary["spanning_probability_se"],
        xerr=None,
        marker="o",
        capsize=3,
        linestyle="-",
    )
    ax.set_xlabel("Actual occupied area fraction")
    ax.set_ylabel("Left-to-right spanning probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Geometric percolation vs actual fraction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_path / "spanning_vs_actual_fraction.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

    # Electrical plots are only produced when resistance data exist.
    if (
        "resistance_ohm" in trial_df.columns
        and trial_df["resistance_ohm"].notna().any()
    ):
        valid = trial_df.dropna(
            subset=["resistance_ohm"]
        ).copy()

        # Plot 3: resistance against nominal fraction.
        grouped = (
            valid.groupby("nominal_fraction", observed=True)
            .agg(
                median_resistance=("resistance_ohm", "median"),
                trial_count=("resistance_ohm", "size"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            grouped["nominal_fraction"],
            grouped["median_resistance"],
            marker="o",
        )
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Nominal disk fraction, η")
        ax.set_ylabel("Median signed transfer resistance (Ω)")
        ax.set_title("Electrical response vs nominal fraction")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_path / "resistance_vs_nominal_fraction.png",
            dpi=300,
        )
        plt.show()
        plt.close(fig)

        # Plot 4: resistance against actual fraction, binned by each
        # morphology's actual occupied fraction.
        if valid["actual_fraction"].nunique() > 1:
            valid["actual_bin"] = pd.cut(
                valid["actual_fraction"],
                bins=actual_fraction_bins,
                include_lowest=True,
                duplicates="drop",
            )

            resistance_actual = (
                valid.groupby("actual_bin", observed=True)
                .agg(
                    mean_actual_fraction=("actual_fraction", "mean"),
                    median_resistance=("resistance_ohm", "median"),
                    trial_count=("resistance_ohm", "size"),
                )
                .reset_index()
            )
        else:
            resistance_actual = (
                valid.groupby("actual_fraction", observed=True)
                .agg(
                    mean_actual_fraction=("actual_fraction", "mean"),
                    median_resistance=("resistance_ohm", "median"),
                    trial_count=("resistance_ohm", "size"),
                )
                .reset_index()
            )

        resistance_actual.to_csv(
            output_path / "resistance_actual_fraction_bins.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            resistance_actual["mean_actual_fraction"],
            resistance_actual["median_resistance"],
            marker="o",
        )
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Actual occupied area fraction")
        ax.set_ylabel("Median signed transfer resistance (Ω)")
        ax.set_title("Electrical response vs actual fraction")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            output_path / "resistance_vs_actual_fraction.png",
            dpi=300,
        )
        plt.show()
        plt.close(fig)

    return actual_summary


if __name__ == "__main__":
    # Initial test settings. Increase trials for final threshold estimates.
    OUTPUT_DIR = "monte_carlo_output_trial_100"

    results, trial_df = run_monte_carlo_sweep(
        nominal_fractions=[
            0.10, 0.20, 0.30, 0.50, 0.60,
            0.70, 0.80, 0.90, 0.95, 1.00,
            1.05, 1.10, 1.15, 1.20, 1.25, 
            1.3, 1.35, 1.4, 1.45, 1.5
        ],
        trials_per_fraction=100,
        width=20.0,
        height=12.0,
        diameter_range=(0.2, 0.6),
        contact_tolerance=0.02,
        seed=42,
        allow_overlap=True,
        probe_spacing=3.0,
        grain_boundary_resistance_ohm=100.0,
        probe_contact_radius=0.15,
        grain_resistivity_ohm_m=1e-4,
        film_thickness_m=92e-9,
        effective_width_m=200e-9,
        area_pixels_per_unit=50,
        output_dir=OUTPUT_DIR,
    )

    plot_monte_carlo_results(
        results,
        trial_df,
        output_dir=OUTPUT_DIR,
        actual_fraction_bins=15,
    )

    print("\nSummary:")
    print(
        pd.DataFrame([result.to_dict() for result in results])[
            [
                "nominal_fraction",
                "mean_actual_fraction",
                "spanning_probability",
                "spanning_probability_se",
                "resistance_valid_fraction",
                "negative_resistance_count",
            ]
        ].to_string(index=False)
    )
