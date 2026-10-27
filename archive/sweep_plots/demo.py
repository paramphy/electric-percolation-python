"""Small demonstration of the CuS 2D percolation prototype.

Install:
    pip install numpy scipy networkx matplotlib

Run from this folder:
    python demo.py

This is a code smoke-test, not a calibrated prediction for the experiment.
Lengths are in micrometres. The default domain is intentionally small.
"""
import matplotlib.pyplot as plt

from morphology import generate_grains, estimate_occupied_fraction
from network import build_contact_graph, find_spanning_cluster
from electrical import four_probe_resistance
from visualization import plot_morphology, plot_contact_network

# Small illustrative domain: 30 x 20 micrometres.
# Probe spacing is reduced here so all four probes fit in the demonstration.
morph = generate_grains(
    width=30.0,
    height=20.0,
    target_fraction=1.8,
    diameter_range=(0.2, 0.6),
    seed=5,
    allow_overlap=True,
)
graph = build_contact_graph(morph)
spanning, cluster = find_spanning_cluster(morph, graph)

print(f"Number of grains: {morph.n_grains}")
print(f"Nominal CuS area fraction: {morph.nominal_fraction:.3f}")
print(f"Number of contact edges: {graph.number_of_edges()}")
print(f"Left-right spanning cluster: {spanning}")

# A 6 micrometre spacing is used only for this demo geometry.
result = four_probe_resistance(
    morph,
    graph,
    probe_spacing=6.0,
    current_amp=1.0,
    grain_boundary_resistance_ohm=100.0,
)
print("Four-probe resistance (demo units):", result["resistance_ohm"])
if result.get("reason"):
    print("Solver note:", result["reason"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_morphology(morph, ax=axes[0])
plot_contact_network(morph, graph, ax=axes[1])
plt.tight_layout()
plt.show()

for i in range(20):
    phi = estimate_occupied_fraction(
        centers=morph.centers,
        radii=morph.radii,
        width=morph.width,
        height=morph.height,
        seed=i
    )
    print(f"Estimated occupied fraction (seed {i}): {phi:.4f}")
    #print(f"Actual occupied fraction: {phi:.4f}")
    print(f"Actual occupied percentage: {100 * phi:.2f}%")