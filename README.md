# Monte Carlo Percolation and Electrical Resistance

A single-core Python workflow to study geometric percolation and four-probe resistance in simulated grain-based thin films.

## Features

- Monte Carlo simulations across four particle-size ranges and eight nominal fractions.
- Geometric spanning probability and percolation-threshold estimates.
- Actual occupied area-fraction calculation.
- Four-probe resistance calculations for spanning morphologies.
- CSV summaries, metadata, and publication-ready PNG/PDF plots.

## Requirements

Python 3.10+ with NumPy, pandas, Matplotlib, and SciPy, plus the project modules:

- `morphology.py`
- `network.py`
- `area_fraction.py`
- `electrical.py`

Install dependencies:

```bash
python -m pip install numpy pandas matplotlib scipy
```

## Run

From the project root:

```bash
python percolation_size_threshold_study_with_resistance.py
```

The default configuration uses 500 trials per condition: 4 particle-size ranges × 8 nominal fractions = 16,000 trials.

## Key Parameters

| Parameter | Default |
|---|---:|
| Grain-boundary resistance | 1 Ω |
| Grain resistivity | 10⁻⁵ Ω·m |
| Film thickness | 92 nm |
| Effective width | 200 nm |
| Probe spacing | 1 model unit |
| Target valid resistance readings | 100 per condition |

Probe spacing and length conversion assume one model unit corresponds to 1 µm. Verify these assumptions, along with film thickness and effective width, before comparing results with experiments.

Only finite, positive resistance values are retained; negative values are rejected, not converted using `abs()`. The target of 100 valid readings is not guaranteed for every condition.

## Outputs

The results directory contains trial-level data, percolation and resistance summaries, threshold estimates, run metadata, and figures in PNG/PDF formats.

Key files include:

- `percolation_trial_results.csv`
- `percolation_threshold_estimates.csv`
- `resistance_summary_by_condition.csv`
- `valid_resistance_measurements.csv`
- `run_metadata.json`

## Notes

- Nominal fraction can exceed 1 because particles may overlap.
- Actual area fraction is estimated by rasterizing the 2D morphology.
- Geometric spanning does not guarantee a valid electrical result.
- Results depend on simulation parameters and electrical-model assumptions; validate them before making quantitative experimental claims.

