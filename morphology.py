"""2D CuS grain morphology for continuum-percolation simulations.

All lengths use one consistent unit, for example micrometres.

With allow_overlap=True, grain centres are distributed uniformly throughout
the domain. Disks may overlap one another and intersect domain boundaries.

The reported nominal_fraction is sum(pi*r**2)/(width*height). It is NOT
the true occupied area fraction when disks overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Morphology:
    centers: FloatArray
    radii: FloatArray
    width: float
    height: float
    target_fraction: float
    nominal_fraction: float

    @property
    def n_grains(self) -> int:
        return int(self.radii.size)


def generate_grains(
    width: float,
    height: float,
    target_fraction: float,
    diameter_range: tuple[float, float] = (0.2, 0.6),
    seed: int | None = None,
    allow_overlap: bool = True,
    max_attempts: int = 2_000_000,
) -> Morphology:
    """Generate random circular grains in a rectangular domain.

    Parameters
    ----------
    width, height:
        Domain dimensions in the same units as diameter_range.

    target_fraction:
        Target nominal disk density:
            sum(pi * radius**2) / (width * height)

        When overlap is allowed, this value can exceed 1.

    diameter_range:
        Uniform diameter range. The current default is a provisional
        approximation to the 200–600 nm features discussed for the films.

    seed:
        Random seed for reproducibility.

    allow_overlap:
        True:
            Standard overlapping-disk continuum-percolation model.

        False:
            Non-overlapping random sequential addition. This approach jams
            at high density and is not suitable for the main density sweep.

    max_attempts:
        Maximum number of placement attempts.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if target_fraction < 0:
        raise ValueError("target_fraction must be non-negative.")

    if not allow_overlap and target_fraction > 1:
        raise ValueError(
            "Fractions above 1 require allow_overlap=True."
        )

    dmin, dmax = diameter_range

    if dmin <= 0 or dmax < dmin:
        raise ValueError(
            "diameter_range must satisfy 0 < min <= max."
        )

    if target_fraction == 0:
        return Morphology(
            centers=np.empty((0, 2), dtype=float),
            radii=np.empty(0, dtype=float),
            width=width,
            height=height,
            target_fraction=0.0,
            nominal_fraction=0.0,
        )

    rng = np.random.default_rng(seed)

    centers: list[tuple[float, float]] = []
    radii: list[float] = []

    nominal_area = 0.0
    domain_area = width * height
    attempts = 0

    while nominal_area / domain_area < target_fraction:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not reach the requested density. "
                "Try increasing max_attempts or check the placement settings."
            )

        radius = float(rng.uniform(dmin / 2, dmax / 2))

        if 2 * radius > min(width, height):
            raise ValueError(
                "A grain is larger than the simulation domain."
            )

        if allow_overlap:
            # Centres can be anywhere in the domain.
            # Disks may overlap and may cross the domain boundary.
            x = float(rng.uniform(0.0, width))
            y = float(rng.uniform(0.0, height))
        else:
            # Keep complete disks inside the domain.
            x = float(rng.uniform(radius, width - radius))
            y = float(rng.uniform(radius, height - radius))

        if not allow_overlap and centers:
            existing_centers = np.asarray(centers)
            existing_radii = np.asarray(radii)

            distances_squared = np.sum(
                (
                    existing_centers
                    - np.array([x, y])
                ) ** 2,
                axis=1,
            )

            if np.any(
                distances_squared
                < (existing_radii + radius) ** 2
            ):
                continue

        centers.append((x, y))
        radii.append(radius)

        # This is nominal disk area. Overlapping areas are counted repeatedly.
        nominal_area += np.pi * radius**2

    centers_array = np.asarray(
        centers, dtype=float
    ).reshape(-1, 2)

    radii_array = np.asarray(radii, dtype=float)

    achieved_nominal_fraction = (
        np.sum(np.pi * radii_array**2) / domain_area
    )

    return Morphology(
        centers=centers_array,
        radii=radii_array,
        width=width,
        height=height,
        target_fraction=target_fraction,
        nominal_fraction=float(achieved_nominal_fraction),
    )

def generate_grains_lognormal(
    width: float,
    height: float,
    target_fraction: float,
    mean_diameter: float = 0.4,
    diameter_cv: float = 0.25,
    diameter_bounds: tuple[float, float] = (0.2, 0.6),
    seed: int | None = None,
    max_attempts: int = 2_000_000,
) -> Morphology:
    """Generate overlapping circular grains with a lognormal size distribution.

    Parameters
    ----------
    width, height:
        Simulation domain dimensions.

    target_fraction:
        Target nominal disk density sum(pi*r**2)/(width*height).
        Overlapping areas are counted repeatedly.

    mean_diameter:
        Target arithmetic mean diameter, in the same units as width/height.

    diameter_cv:
        Target coefficient of variation (standard deviation / mean).
        Must be non-negative. CV=0 gives monodisperse grains.

    diameter_bounds:
        (minimum, maximum) diameter. Values are truncated to these bounds,
        so the realized mean and CV may differ from the requested values.

    seed:
        Random seed.

    max_attempts:
        Maximum number of generated grains.

    Notes
    -----
    The generator stops when the sum of disk areas reaches or exceeds the
    target nominal disk density. The final achieved density may slightly
    exceed the target because grains are added as whole disks.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if target_fraction < 0:
        raise ValueError("target_fraction must be non-negative.")

    if mean_diameter <= 0:
        raise ValueError("mean_diameter must be positive.")

    if diameter_cv < 0:
        raise ValueError("diameter_cv must be non-negative.")

    dmin, dmax = diameter_bounds

    if dmin <= 0 or dmax < dmin:
        raise ValueError("Invalid diameter_bounds.")

    if not dmin <= mean_diameter <= dmax:
        raise ValueError(
            "mean_diameter must lie inside diameter_bounds."
        )

    if dmax > min(width, height):
        raise ValueError(
            "Maximum diameter must not exceed the smaller domain dimension."
        )

    rng = np.random.default_rng(seed)

    if target_fraction == 0:
        return Morphology(
            centers=np.empty((0, 2), dtype=float),
            radii=np.empty(0, dtype=float),
            width=width,
            height=height,
            target_fraction=0.0,
            nominal_fraction=0.0,
        )

    domain_area = width * height
    centers: list[tuple[float, float]] = []
    radii: list[float] = []
    nominal_area = 0.0

    # Convert arithmetic mean and CV into lognormal parameters.
    if diameter_cv == 0:
        sigma_log = 0.0
        mu_log = np.log(mean_diameter)
    else:
        sigma_log = np.sqrt(
            np.log1p(diameter_cv**2)
        )
        mu_log = (
            np.log(mean_diameter)
            - 0.5 * sigma_log**2
        )

    attempts = 0

    while nominal_area / domain_area < target_fraction:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Could not reach target nominal density before "
                "max_attempts. Increase max_attempts or reduce density."
            )

        if diameter_cv == 0:
            diameter = mean_diameter
        else:
            # Rejection sampling preserves the lognormal distribution
            # conditional on the requested diameter bounds.
            for _ in range(1000):
                candidate = float(
                    rng.lognormal(
                        mean=mu_log,
                        sigma=sigma_log,
                    )
                )
                if dmin <= candidate <= dmax:
                    diameter = candidate
                    break
            else:
                raise RuntimeError(
                    "Unable to sample a diameter within the bounds. "
                    "Check mean, CV and diameter_bounds."
                )

        radius = diameter / 2.0

        # Overlapping continuum-percolation geometry: centres are uniform
        # throughout the domain; disks can intersect its boundaries.
        x = float(rng.uniform(0.0, width))
        y = float(rng.uniform(0.0, height))

        centers.append((x, y))
        radii.append(radius)
        nominal_area += np.pi * radius**2

    centers_array = np.asarray(
        centers,
        dtype=float,
    ).reshape(-1, 2)

    radii_array = np.asarray(radii, dtype=float)

    return Morphology(
        centers=centers_array,
        radii=radii_array,
        width=width,
        height=height,
        target_fraction=target_fraction,
        nominal_fraction=float(
            np.sum(np.pi * radii_array**2) / domain_area
        ),
    )


import numpy as np


def estimate_occupied_fraction(
    centers,
    radii,
    width,
    height,
    n_points=1_000_000,
    seed=10,
):
    """
    Estimate the fraction of a rectangular domain covered
    by the union of circular grains.

    Disk portions outside the domain are excluded.
    Overlapping regions are counted only once.

    centers: array of shape (N, 2), in domain length units
    radii: array of shape (N,), in the same units
    width, height: domain dimensions
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    rng = np.random.default_rng(seed)

    points = np.column_stack([
        rng.uniform(0, width, n_points),
        rng.uniform(0, height, n_points),
    ])

    covered = np.zeros(n_points, dtype=bool)

    # Process one disk at a time to limit memory usage.
    for (cx, cy), radius in zip(centers, radii):
        if radius <= 0:
            continue

        inside = (
            (points[:, 0] - cx) ** 2
            + (points[:, 1] - cy) ** 2
            <= radius ** 2
        )
        covered |= inside

        if covered.all():
            return 1.0

    return float(covered.mean())
