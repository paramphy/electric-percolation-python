"""Contact graph and spanning-cluster analysis for 2D CuS grains."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from morphology import Morphology


def build_contact_graph(
    morphology: Morphology,
    contact_tolerance: float = 0.0,
) -> nx.Graph:
    """Create a grain graph; edges represent touching/near-touching circles.

    contact_tolerance=0 means geometrical contact only. A positive value is
    a geometrical tolerance, NOT a tunnelling model.
    """
    if contact_tolerance < 0:
        raise ValueError("contact_tolerance must be non-negative.")
    graph = nx.Graph()
    for i, (center, radius) in enumerate(zip(morphology.centers, morphology.radii)):
        graph.add_node(
            i, x=float(center[0]), y=float(center[1]), radius=float(radius)
        )

    if morphology.n_grains < 2:
        return graph

    max_radius = float(np.max(morphology.radii))
    pairs = cKDTree(morphology.centers).query_pairs(
        r=2 * max_radius + contact_tolerance
    )
    for i, j in pairs:
        delta = morphology.centers[i] - morphology.centers[j]
        distance = float(np.linalg.norm(delta))
        gap = distance - (morphology.radii[i] + morphology.radii[j])
        if gap <= contact_tolerance:
            graph.add_edge(i, j, gap=max(0.0, gap))
    return graph


def find_spanning_cluster(
    morphology: Morphology,
    graph: nx.Graph,
    electrode_margin: float = 0.0,
) -> tuple[bool, set[int]]:
    """Find grains touching both left and right boundaries.

    The four-probe electrical solver uses explicit collinear probe contacts;
    this separate check is a simple left-to-right geometric spanning test.
    """
    left = {
        i for i, x in enumerate(morphology.centers[:, 0])
        if x - morphology.radii[i] <= electrode_margin
    }
    right = {
        i for i, x in enumerate(morphology.centers[:, 0])
        if x + morphology.radii[i] >= morphology.width - electrode_margin
    }
    for component in nx.connected_components(graph):
        comp = set(component)
        if comp & left and comp & right:
            return True, comp
    return False, set()
