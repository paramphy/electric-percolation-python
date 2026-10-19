"""Monte Carlo sweeps over CuS grain-size distributions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from morphology import generate_grains_lognormal
from network import build_contact_graph, find_spanning_cluster
from electrical import four_probe_resistance_multiterminal


@dataclass
class SizeSweepResult:
    sweep_type: str
    nominal_fraction: float
    requested_mean_diameter: float
    requested_cv: float
    trials: int

    mean_realized_diameter: float
    mean_realized_cv: float
    mean_grain_count: float
    mean_achieved_nominal_fraction: float

    spanning_count: int
    spanning_probability: float
    spanning_probability_se: float

    valid_resistance_count: int
    resistance_valid_fraction: float
    positive_resistance_count: int
    negative_resistance_count: int

    median_positive_resistance_ohm: float | None
    mean_positive_resistance_ohm: float | None
    q25_positive_resistance_ohm: float | None
    q75_positive_resistance_ohm: float | None

    signed_resistance_values_ohm: list[float]


def run_size_distribution_sweep(
    *,
    sweep_type: str,
    values: list[float],
    nominal_fraction: float,
    trials_per_setting: int = 100,
    width: float = 20.0,
    height: float = 12.0,
    baseline_mean_diameter: float = 0.4,
    baseline_cv: float = 0.25,
    diameter_bounds: tuple[float, float] = (0.1, 1.0),
    contact_tolerance: float = 0.0,
    probe_spacing: float = 3.0,
    current_amp: float = 1.0,
    grain_boundary_resistance_ohm: float = 100.0,
    grain_resistivity_ohm_m: float = 1e-5,
    film_thickness_m: float = 92e-9,
    effective_width_m: float = 200e-9,
    probe_contact_radius: float = 0.15,
    seed: int = 1234,
) -> list[SizeSweepResult]:
    """Run a particle-size distribution sweep.

    sweep_type:
        "cv"   -> vary distribution CV, hold mean diameter fixed.
        "mean" -> vary mean diameter, hold CV fixed.

    All geometry dimensions and diameters use the same length unit.
    """
    if sweep_type not in {"cv", "mean"}:
        raise ValueError("sweep_type must be 'cv' or 'mean'.")

    if trials_per_setting < 1:
        raise ValueError("trials_per_setting must be >= 1.")

    if not values:
        raise ValueError("values must not be empty.")

    if probe_spacing <= 0:
        raise ValueError("probe_spacing must be positive.")

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(
        seed_sequence.spawn(len(values) * trials_per_setting)
    )

    results: list[SizeSweepResult] = []

    for value in values:
        if sweep_type == "cv":
            mean_diameter = baseline_mean_diameter
            cv = float(value)
        else:
            mean_diameter = float(value)
            cv = baseline_cv

        spanning_flags: list[bool] = []
        signed_resistances: list[float] = []
        realized_means: list[float] = []
        realized_cvs: list[float] = []
        grain_counts: list[int] = []
        achieved_fractions: list[float] = []

        for _ in range(trials_per_setting):
            trial_seed = int(
                next(child_seeds).generate_state(1)[0]
            )

            morphology = generate_grains_lognormal(
                width=width,
                height=height,
                target_fraction=nominal_fraction,
                mean_diameter=mean_diameter,
                diameter_cv=cv,
                diameter_bounds=diameter_bounds,
                seed=trial_seed,
            )

            diameters = 2.0 * morphology.radii

            realized_means.append(float(np.mean(diameters)))
            realized_cvs.append(
                float(np.std(diameters) / np.mean(diameters))
            )
            grain_counts.append(morphology.n_grains)
            achieved_fractions.append(morphology.nominal_fraction)

            graph = build_contact_graph(
                morphology,
                contact_tolerance=contact_tolerance,
            )

            spans, _ = find_spanning_cluster(
                morphology,
                graph,
            )
            spanning_flags.append(bool(spans))

            electrical = four_probe_resistance_multiterminal(
                morphology=morphology,
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

            # Preserve signed finite values for diagnostics.
            # Invalid/disconnected cases return infinity and are excluded.
            if np.isfinite(resistance) and electrical.get(
                "percolates", False
            ):
                signed_resistances.append(resistance)

        spanning_count = int(np.count_nonzero(spanning_flags))
        probability = spanning_count / trials_per_setting
        probability_se = float(
            np.sqrt(
                probability * (1.0 - probability)
                / trials_per_setting
            )
        )

        positive = [
            r for r in signed_resistances if r > 0
        ]
        negative_count = sum(
            r < 0 for r in signed_resistances
        )

        if positive:
            q25, median, q75 = np.percentile(
                positive,
                [25, 50, 75],
            )
            mean_positive = float(np.mean(positive))
            median_positive = float(median)
            q25_positive = float(q25)
            q75_positive = float(q75)
        else:
            mean_positive = None
            median_positive = None
            q25_positive = None
            q75_positive = None

        results.append(
            SizeSweepResult(
                sweep_type=sweep_type,
                nominal_fraction=nominal_fraction,
                requested_mean_diameter=mean_diameter,
                requested_cv=cv,
                trials=trials_per_setting,
                mean_realized_diameter=float(np.mean(realized_means)),
                mean_realized_cv=float(np.mean(realized_cvs)),
                mean_grain_count=float(np.mean(grain_counts)),
                mean_achieved_nominal_fraction=float(
                    np.mean(achieved_fractions)
                ),
                spanning_count=spanning_count,
                spanning_probability=float(probability),
                spanning_probability_se=probability_se,
                valid_resistance_count=len(signed_resistances),
                resistance_valid_fraction=(
                    len(signed_resistances) / trials_per_setting
                ),
                positive_resistance_count=len(positive),
                negative_resistance_count=negative_count,
                median_positive_resistance_ohm=median_positive,
                mean_positive_resistance_ohm=mean_positive,
                q25_positive_resistance_ohm=q25_positive,
                q75_positive_resistance_ohm=q75_positive,
                signed_resistance_values_ohm=signed_resistances,
            )
        )

        print(
            f"{sweep_type}={value:.3f} | "
            f"spanning={spanning_count}/{trials_per_setting} | "
            f"valid R={len(signed_resistances)}/{trials_per_setting} | "
            f"negative R={negative_count}"
        )

    return results


def save_sweep_csv(
    results: list[SizeSweepResult],
    filename: str,
) -> None:
    """Save summary rows to CSV, excluding individual resistance lists."""
    rows = []

    for result in results:
        row = asdict(result)
        row.pop("signed_resistance_values_ohm", None)
        rows.append(row)

    if not rows:
        raise ValueError("No results to save.")

    path = Path(filename)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    mean_results = run_size_distribution_sweep(
        sweep_type="mean",

        # Characteristic particle size, in micrometres
        values=[
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            1.0,
        ],

        # Hold nominal disk density fixed
        nominal_fraction=1.2,

        # Increase to 100–200 after confirming runtime
        trials_per_setting=200,

        width=20.0,
        height=12.0,

        # Keep the distribution narrow and fixed
        baseline_mean_diameter=0.4,
        baseline_cv=0.05,
        diameter_bounds=(0.1, 2),

        contact_tolerance=0.02,
        probe_spacing=3.0,

        grain_boundary_resistance_ohm=1000.0,
        grain_resistivity_ohm_m=1e-4,
        film_thickness_m=92e-9,
        effective_width_m=200e-9,

        probe_contact_radius=0.15,
        seed=202,
    )

    save_sweep_csv(
        mean_results,
        "particle_size_characteristic_sweep.csv",
    )