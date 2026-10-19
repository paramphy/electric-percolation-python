
# area_fraction.py

import numpy as np


def rasterized_occupied_fraction(
    centers,
    radii,
    width,
    height,
    pixels_per_unit=50,
):
    """
    Estimate the actual area fraction covered by circular grains.

    Parameters
    ----------
    centers : array-like, shape (N, 2)
        Grain centres as (x, y), in the same units as width/height.
    radii : array-like, shape (N,)
        Grain radii in the same units as width/height.
    width, height : float
        Rectangular simulation-domain dimensions.
    pixels_per_unit : float
        Raster resolution. For example, 50 means 50 pixels per
        micrometre if the domain dimensions are in micrometres.

    Returns
    -------
    float
        Estimated union area divided by domain area.

    Notes
    -----
    - Overlapping disk pixels are counted only once.
    - Disk portions outside the domain are excluded.
    - Pixels are classified by testing their centres.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    if pixels_per_unit <= 0:
        raise ValueError("pixels_per_unit must be positive.")

    if centers.size == 0:
        return 0.0

    centers = centers.reshape(-1, 2)

    if len(centers) != len(radii):
        raise ValueError("centers and radii must have equal lengths.")

    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain finite values.")

    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("radii must be finite and non-negative.")

    # Choose integer grid dimensions so every pixel has the same area.
    nx = max(1, int(np.ceil(width * pixels_per_unit)))
    ny = max(1, int(np.ceil(height * pixels_per_unit)))

    dx = width / nx
    dy = height / ny

    # Pixel-centre coordinates, in domain units.
    x_pixels = (np.arange(nx) + 0.5) * dx
    y_pixels = (np.arange(ny) + 0.5) * dy

    # True means that this pixel centre lies inside at least one disk.
    covered = np.zeros((ny, nx), dtype=bool)

    for (cx, cy), radius in zip(centers, radii):
        if radius == 0:
            continue

        # Restrict calculations to this disk's bounding box,
        # clipped to the simulation domain.
        ix0 = max(0, int(np.floor((cx - radius) / dx)))
        ix1 = min(nx, int(np.ceil((cx + radius) / dx)))
        iy0 = max(0, int(np.floor((cy - radius) / dy)))
        iy1 = min(ny, int(np.ceil((cy + radius) / dy)))

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        xs = x_pixels[ix0:ix1]
        ys = y_pixels[iy0:iy1]

        distance_squared = (
            (xs[np.newaxis, :] - cx) ** 2
            + (ys[:, np.newaxis] - cy) ** 2
        )

        covered[iy0:iy1, ix0:ix1] |= (
            distance_squared <= radius**2
        )

        # Once the entire domain is covered, no further work is needed.
        if covered.all():
            return 1.0

    return float(covered.mean())


def actual_fraction_from_morphology(
    morphology,
    pixels_per_unit=50,
):
    """Convenience wrapper for the project's Morphology object."""
    return rasterized_occupied_fraction(
        centers=morphology.centers,
        radii=morphology.radii,
        width=morphology.width,
        height=morphology.height,
        pixels_per_unit=pixels_per_unit,
    )
