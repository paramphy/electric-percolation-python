
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load data
# --------------------------------------------------
CSV_FILE = Path("particle_size_characteristic_sweep.csv")
OUTPUT_DIR = Path("sweep_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_FILE)

print("Available columns:", df.columns.tolist())


def find_column(candidates):
    """Return the first matching column name, ignoring case."""
    lookup = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# Find the mean grain diameter column.
x_col = find_column([
    "requested_mean_diameter",
    "mean_diameter",
    "target_mean_diameter",
    "baseline_mean_diameter",
])

if x_col is None:
    raise ValueError(
        "Could not identify the mean diameter column. "
        "Check the printed column names and update x_col."
    )

# --------------------------------------------------
# Plot 1: Spanning probability
# --------------------------------------------------
span_col = find_column([
    "spanning_probability",
    "span_probability",
    "spanning_prob",
])

if span_col:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df[x_col], df[span_col],
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Mean grain diameter (µm)")
    ax.set_ylabel("Spanning probability")
    ax.set_title("Geometric connectivity vs grain size")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "spanning_probability.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

# --------------------------------------------------
# Plot 2: Median positive resistance
# --------------------------------------------------
res_col = find_column([
    "median_positive_resistance_ohm",
    "median_positive_resistance",
    "median_resistance_ohm",
])

q25_col = find_column([
    "q25_positive_resistance_ohm",
    "q25_positive_resistance",
])

q75_col = find_column([
    "q75_positive_resistance_ohm",
    "q75_positive_resistance",
])

if res_col:
    fig, ax = plt.subplots(figsize=(7, 5))

    x = df[x_col]
    y = df[res_col]

    ax.plot(
        x, y,
        marker="o",
        linewidth=2,
        label="Median positive resistance",
    )

    if q25_col and q75_col:
        ax.fill_between(
            x,
            df[q25_col],
            df[q75_col],
            alpha=0.2,
            label="25th–75th percentile",
        )

    ax.set_xlabel("Mean grain diameter (µm)")
    ax.set_ylabel("Four-probe resistance (Ω)")
    ax.set_title("Electrical resistance vs grain size")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Use a logarithmic axis if all plotted values are positive.
    if (y.dropna() > 0).all() and len(y.dropna()) > 0:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "resistance_vs_grain_size.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

# --------------------------------------------------
# Plot 3: Fraction of electrically valid trials
# --------------------------------------------------
valid_col = find_column([
    "valid_resistance_fraction",
    "valid_electrical_fraction",
    "valid_fraction",
])

if valid_col:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df[x_col], df[valid_col],
        marker="s",
        linewidth=2,
    )

    ax.set_xlabel("Mean grain diameter (µm)")
    ax.set_ylabel("Valid resistance fraction")
    ax.set_title("Electrical solver success vs grain size")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "valid_trials_vs_grain_size.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

# --------------------------------------------------
# Plot 4: Number of grains
# --------------------------------------------------
count_col = find_column([
    "mean_grain_count",
    "grain_count",
    "mean_n_grains",
])

if count_col:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df[x_col], df[count_col],
        marker="^",
        linewidth=2,
    )

    ax.set_xlabel("Mean grain diameter (µm)")
    ax.set_ylabel("Number of grains")
    ax.set_title("Grain count vs characteristic grain size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "grain_count_vs_grain_size.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig)

print(f"\nPlots saved in: {OUTPUT_DIR.resolve()}")
