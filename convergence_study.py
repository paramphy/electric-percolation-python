
"""
Monte Carlo convergence study for geometric percolation.

Runs independent morphology realizations at one fixed nominal disk fraction,
then evaluates the cumulative left-to-right spanning probability at selected
trial-count checkpoints.

Outputs:
    results/convergence_trial_results.csv
    results/convergence_checkpoint_summary.csv
    results/convergence_spanning_probability.png
    results/convergence_confidence_interval.png
    results/convergence_metadata.json

Run from the project root:
    python convergence_study.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morphology import generate_grains
from network import build_contact_graph, find_spanning_cluster


# ============================================================
# Configuration: adjust these parameters as needed
# ============================================================

N_TRIALS = 5000
CHECKPOINTS = [50, 100, 200, 500, 1000, 1500, 2000, 5000]

# Choose a fraction near the approximate percolation transition.
# This is nominal disk fraction, not actual occupied area fraction.
NOMINAL_FRACTION = 1.25

DOMAIN_WIDTH = 20.0
DOMAIN_HEIGHT = 12.0
DIAMETER_RANGE = (0.2, 0.6)
CONTACT_TOLERANCE = 0.0

MASTER_SEED = 20261027

OUTPUT_DIR = Path("results") / "convergence_study"


# ============================================================
# Statistics
# ============================================================

def wilson_interval(
    successes: int,
    trials: int,
    confidence_z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive.")

    p_hat = successes / trials
    z = confidence_z
    denominator = 1.0 + z**2 / trials

    centre = (
        p_hat + z**2 / (2.0 * trials)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            p_hat * (1.0 - p_hat) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


# ============================================================
# Simulation
# ============================================================

def run_convergence_study() -> pd.DataFrame:
    """Run the trials and return one row per realization."""
    if N_TRIALS <= 0:
        raise ValueError("N_TRIALS must be positive.")

    if not CHECKPOINTS:
        raise ValueError("CHECKPOINTS cannot be empty.")

    if any(n <= 0 or n > N_TRIALS for n in CHECKPOINTS):
        raise ValueError(
            "Every checkpoint must be between 1 and N_TRIALS."
        )

    if any(
        CHECKPOINTS[i] >= CHECKPOINTS[i + 1]
        for i in range(len(CHECKPOINTS) - 1)
    ):
        raise ValueError("CHECKPOINTS must be strictly increasing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Independent reproducible seed for every trial.
    seed_sequence = np.random.SeedSequence(MASTER_SEED)
    trial_seed_sequences = seed_sequence.spawn(N_TRIALS)

    rows = []
    start_time = time.time()

    for trial_number, seed_sequence in enumerate(
        trial_seed_sequences, start=1
    ):
        trial_seed = int(
            seed_sequence.generate_state(1, dtype=np.uint32)[0]
        )

        morphology = generate_grains(
            width=DOMAIN_WIDTH,
            height=DOMAIN_HEIGHT,
            target_fraction=NOMINAL_FRACTION,
            diameter_range=DIAMETER_RANGE,
            seed=trial_seed,
            allow_overlap=True,
        )

        graph = build_contact_graph(
            morphology,
            contact_tolerance=CONTACT_TOLERANCE,
        )

        # This follows the return convention used by the project version
        # tested for this script: (spans, spanning_cluster).
        spans, cluster = find_spanning_cluster(morphology, graph)

        rows.append(
            {
                "trial": trial_number,
                "seed": trial_seed,
                "nominal_fraction": NOMINAL_FRACTION,
                "grain_count": morphology.n_grains,
                "contact_edges": graph.number_of_edges(),
                "spans_left_to_right": bool(spans),
                "spanning_cluster_grains": (
                    len(cluster) if spans else 0
                ),
            }
        )

        if (
            trial_number % 50 == 0
            or trial_number == N_TRIALS
        ):
            elapsed = time.time() - start_time
            print(
                f"Completed {trial_number}/{N_TRIALS} trials "
                f"({elapsed:.1f} s)"
            )

    trial_df = pd.DataFrame(rows)

    trial_path = OUTPUT_DIR / "convergence_trial_results.csv"
    trial_df.to_csv(trial_path, index=False)

    print(f"\nSaved trial results: {trial_path}")

    return trial_df


# ============================================================
# Checkpoint summary
# ============================================================

def summarize_checkpoints(
    trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate spanning probability and uncertainty at each checkpoint."""
    rows = []
    outcomes = trial_df["spans_left_to_right"].astype(int).to_numpy()

    for checkpoint in CHECKPOINTS:
        first_trials = outcomes[:checkpoint]
        successes = int(first_trials.sum())
        probability = successes / checkpoint
        lower, upper = wilson_interval(successes, checkpoint)

        rows.append(
            {
                "trials": checkpoint,
                "spanning_count": successes,
                "spanning_probability": probability,
                "wilson_95_ci_lower": lower,
                "wilson_95_ci_upper": upper,
                "wilson_95_ci_half_width": (
                    upper - lower
                ) / 2.0,
                "mean_grain_count": trial_df.iloc[
                    :checkpoint
                ]["grain_count"].mean(),
                "mean_contact_edges": trial_df.iloc[
                    :checkpoint
                ]["contact_edges"].mean(),
            }
        )

    summary_df = pd.DataFrame(rows)

    # Difference between cumulative estimates at adjacent checkpoints.
    summary_df["change_vs_previous_checkpoint_pp"] = (
        summary_df["spanning_probability"].diff() * 100.0
    )

    summary_path = (
        OUTPUT_DIR / "convergence_checkpoint_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved checkpoint summary: {summary_path}")

    return summary_df


# ============================================================
# Plots
# ============================================================

def plot_spanning_probability(
    trial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Plot cumulative spanning probability with checkpoint intervals."""
    outcomes = trial_df["spans_left_to_right"].astype(int).to_numpy()
    trial_counts = np.arange(1, len(outcomes) + 1)
    cumulative_probability = (
        np.cumsum(outcomes) / trial_counts
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        trial_counts,
        cumulative_probability,
        linewidth=1.0,
        label="Cumulative spanning probability",
    )

    # Show confidence intervals at each requested checkpoint.
    for _, row in summary_df.iterrows():
        n = int(row["trials"])
        p = float(row["spanning_probability"])

        ax.errorbar(
            n,
            p,
            yerr=[
                [p - row["wilson_95_ci_lower"]],
                [row["wilson_95_ci_upper"] - p],
            ],
            fmt="o",
            capsize=4,
            markersize=4,
        )

    ax.set_xlabel("Cumulative number of independent trials")
    ax.set_ylabel("Estimated left-to-right spanning probability")
    ax.set_title(
        "Monte Carlo convergence "
        f"(nominal disk fraction = {NOMINAL_FRACTION:.2f})"
    )
    ax.set_xlim(1, N_TRIALS)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "convergence_spanning_probability.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {path}")


def plot_confidence_interval(
    summary_df: pd.DataFrame,
) -> None:
    """Plot the 95% confidence-interval half-width against trial count."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(
        summary_df["trials"],
        summary_df["wilson_95_ci_half_width"] * 100.0,
        marker="o",
    )

    ax.set_xlabel("Cumulative number of independent trials")
    ax.set_ylabel("95% confidence interval half-width (percentage points)")
    ax.set_title("Uncertainty in estimated spanning probability")
    ax.set_xscale("log")
    ax.set_xticks(
        CHECKPOINTS,
        labels=[str(n) for n in CHECKPOINTS],
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    path = OUTPUT_DIR / "convergence_confidence_interval.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Starting Monte Carlo convergence study")
    print(f"Nominal fraction: {NOMINAL_FRACTION}")
    print(f"Maximum trials: {N_TRIALS}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Master seed: {MASTER_SEED}\n")

    trial_df = run_convergence_study()
    summary_df = summarize_checkpoints(trial_df)

    plot_spanning_probability(trial_df, summary_df)
    plot_confidence_interval(summary_df)

    metadata = {
        "nominal_fraction": NOMINAL_FRACTION,
        "domain_width": DOMAIN_WIDTH,
        "domain_height": DOMAIN_HEIGHT,
        "diameter_range": DIAMETER_RANGE,
        "contact_tolerance": CONTACT_TOLERANCE,
        "allow_overlap": True,
        "master_seed": MASTER_SEED,
        "maximum_trials": N_TRIALS,
        "checkpoints": CHECKPOINTS,
        "measurement": "left-to-right geometric spanning probability",
        "area_fraction_note": (
            "NOMINAL_FRACTION is the nominal sum of disk areas divided "
            "by domain area; it is not the actual union area fraction."
        ),
    }

    metadata_path = OUTPUT_DIR / "convergence_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCheckpoint summary:")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
