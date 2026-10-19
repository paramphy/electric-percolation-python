"""Plots for morphology, connectivity and four-probe network results."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx

from morphology import Morphology
from network import find_spanning_cluster


def plot_morphology(morphology: Morphology, ax=None, probes=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    for (x, y), r in zip(morphology.centers, morphology.radii):
        ax.add_patch(Circle((x, y), r, fill=False, linewidth=0.7))
    ax.set_xlim(0, morphology.width)
    ax.set_ylim(0, morphology.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (model length units)")
    ax.set_ylabel("y (model length units)")
    ax.set_title(
        f"CuS morphology: N={morphology.n_grains}, "
        f"nominal area fraction={morphology.nominal_fraction:.3f}"
    )
    if probes is not None:
        for i, (x, y) in enumerate(probes):
            ax.plot(x, y, marker="v", markersize=8)
            ax.annotate(f"P{i+1}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center")
    return fig, ax


def plot_contact_network(morphology: Morphology, graph: nx.Graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    spanning, cluster = find_spanning_cluster(morphology, graph)
    for u, v in graph.edges():
        x1, y1 = morphology.centers[u]
        x2, y2 = morphology.centers[v]
        ax.plot([x1, x2], [y1, y2], linewidth=0.7, alpha=0.65)
    for i, ((x, y), r) in enumerate(zip(morphology.centers, morphology.radii)):
        ax.add_patch(Circle((x, y), r, fill=False,
                            linewidth=1.0 if i in cluster else 0.5,
                            alpha=1.0 if i in cluster else 0.35))
    ax.set_xlim(0, morphology.width)
    ax.set_ylim(0, morphology.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (model length units)")
    ax.set_ylabel("y (model length units)")
    ax.set_title("Contact network — " + ("spanning cluster found" if spanning else "no left-right spanning cluster"))
    return fig, ax


def plot_voltage_network(result: dict, ax=None):
    if not result.get("percolates") or "graph" not in result:
        raise ValueError("A valid solved network is required.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    graph = result["graph"]
    voltages = result["voltages"]
    probe_nodes = ["P0", "P1", "P2", "P3"]
    # Probe voltage values are known. Grain voltages are not returned in this
    # minimal first version; plot probe voltages as a diagnostic.
    ax.plot(range(1, 5), voltages, marker="o")
    ax.set_xticks(range(1, 5), ["P1", "P2", "P3", "P4"])
    ax.set_ylabel("Potential (V)")
    ax.set_title("Four-probe node potentials")
    ax.grid(True, alpha=0.3)
    return fig, ax
