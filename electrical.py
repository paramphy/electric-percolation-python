"""Stable resistor-network solver for four collinear point probes.

Model:
- Each CuS grain is an equipotential node.
- Each grain-grain contact is a resistor R_gb.
- Probe contacts are ideal electrical connections, implemented by node
  contraction rather than extremely small artificial resistors.
- P0 injects current, P3 extracts current.
- P1 and P2 are high-impedance voltage probes.

Lengths must use a consistent unit.

Important:
The model resolves grain-boundary resistance only. It does not resolve
resistance inside individual grains.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from morphology import Morphology


def probe_positions(
    width: float,
    spacing: float,
    y: float | None = None,
) -> np.ndarray:
    """Return the four equally spaced probe positions."""
    if spacing <= 0:
        raise ValueError("Probe spacing must be positive.")
    if width < 3 * spacing:
        raise ValueError(
            "Domain width must be at least three probe spacings."
        )

    if y is None:
        y = 0.5

    start = 0.5 * (width - 3 * spacing)

    return np.column_stack(
        (
            start + spacing * np.arange(4),
            np.full(4, y),
        )
    )

def calculate_grain_resistances(
    morphology: Morphology,
    grain_resistivity_ohm_m: float,
    film_thickness_m: float,
    effective_width_m: float,
    length_unit_to_m: float = 1e-6,
) -> dict[int, float]:
    """Estimate effective intragrain resistance for every grain.

    Uses:
        R_grain = rho_grain * (2 * radius) / (thickness * width)

    Parameters
    ----------
    grain_resistivity_ohm_m:
        Effective grain resistivity in ohm metres.

    film_thickness_m:
        Film thickness in metres.

    effective_width_m:
        Assumed conducting width in metres. It is held fixed, so larger
        grains have a longer effective current path and higher resistance.

    length_unit_to_m:
        Conversion from morphology length units to metres.
        For micrometre coordinates, use 1e-6.

    Returns
    -------
    Dictionary mapping grain index to effective grain resistance in ohms.
    """
    if not np.isfinite(grain_resistivity_ohm_m) or grain_resistivity_ohm_m < 0:
        raise ValueError(
            "Grain resistivity must be finite and non-negative."
        )

    if not np.isfinite(film_thickness_m) or film_thickness_m <= 0:
        raise ValueError("Film thickness must be finite and positive.")

    if not np.isfinite(effective_width_m) or effective_width_m <= 0:
        raise ValueError("Effective width must be finite and positive.")

    if not np.isfinite(length_unit_to_m) or length_unit_to_m <= 0:
        raise ValueError("length_unit_to_m must be finite and positive.")

    radii_m = np.asarray(morphology.radii, dtype=float) * length_unit_to_m

    grain_resistances = (
        grain_resistivity_ohm_m
        * (2.0 * radii_m)
        / (film_thickness_m * effective_width_m)
    )

    return {
        i: float(resistance)
        for i, resistance in enumerate(grain_resistances)
    }

def _contact_grains(
    morphology: Morphology,
    graph: nx.Graph,
    px: float,
    py: float,
    contact_radius: float,
) -> list[int]:
    """Return grains geometrically touched by a probe.

    Grain node labels are assumed to correspond to indices in morphology.
    """
    if contact_radius < 0:
        raise ValueError("contact_radius must be non-negative.")

    contacted = []

    for i, (x, y) in enumerate(morphology.centers):
        if i not in graph:
            continue

        distance = np.hypot(x - px, y - py)

        if distance <= morphology.radii[i] + contact_radius:
            contacted.append(i)

    return contacted


def solve_resistor_network(
    graph: nx.Graph,
    source,
    sink,
    current_amp: float = 1.0,
    resistance_attribute: str = "resistance",
) -> dict:
    """Solve a positive-resistance graph with current injection at source.

    The sink is grounded. Only the source-sink connected component is solved.
    Disconnected networks, invalid resistances, and failed numerical solutions
    are reported explicitly.
    """
    if not np.isfinite(current_amp) or current_amp <= 0:
        raise ValueError("current_amp must be finite and positive.")

    if source not in graph or sink not in graph:
        return {
            "connected": False,
            "voltages": {},
            "reason": "Source or sink is absent.",
        }

    if source == sink:
        return {
            "connected": False,
            "voltages": {},
            "reason": "Source and sink are electrically shorted.",
        }

    if not nx.has_path(graph, source, sink):
        return {
            "connected": False,
            "voltages": {},
            "reason": "No conductive path between source and sink.",
        }

    component = nx.node_connected_component(graph, source)
    nodes = list(component)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    laplacian = lil_matrix((n, n), dtype=float)

    for u, v, data in graph.subgraph(component).edges(data=True):
        resistance = float(
            data.get(resistance_attribute, np.nan)
        )

        if not np.isfinite(resistance) or resistance <= 0:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) must have finite, "
                "positive resistance."
            )

        conductance = 1.0 / resistance
        i = node_to_idx[u]
        j = node_to_idx[v]

        laplacian[i, i] += conductance
        laplacian[j, j] += conductance
        laplacian[i, j] -= conductance
        laplacian[j, i] -= conductance

    sink_idx = node_to_idx[sink]
    source_idx = node_to_idx[source]

    solve_indices = [
        i for i in range(n) if i != sink_idx
    ]

    rhs = np.zeros(n, dtype=float)
    rhs[source_idx] = current_amp
    rhs[sink_idx] = -current_amp

    reduced_laplacian = (
        laplacian.tocsr()[solve_indices, :][:, solve_indices]
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)

            solution = spsolve(
                reduced_laplacian,
                rhs[solve_indices],
            )

    except (
        MatrixRankWarning,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return {
            "connected": False,
            "voltages": {},
            "reason": f"Network solve failed: {exc}",
        }

    if not np.all(np.isfinite(solution)):
        return {
            "connected": False,
            "voltages": {},
            "reason": "Network solve returned non-finite voltages.",
        }

    voltages = {sink: 0.0}

    for local_idx, global_idx in enumerate(solve_indices):
        voltages[nodes[global_idx]] = float(
            solution[local_idx]
        )

    return {
        "connected": True,
        "voltages": voltages,
        "reason": None,
    }


class _UnionFind:
    """Minimal disjoint-set structure for ideal electrical connections."""

    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, node):
        parent = self.parent[node]

        if parent != node:
            self.parent[node] = self.find(parent)

        return self.parent[node]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _contract_ideal_probe_contacts(
    graph: nx.Graph,
    probe_nodes: list[str],
    contacted: list[list[int]],
    grain_boundary_resistance_ohm: float,
) -> nx.Graph:
    """Contract probe-to-grain connections into ideal electrical nodes.

    If contraction produces multiple grain-boundary edges between the same
    two electrical nodes, they are combined as parallel resistors.
    """
    nodes = list(graph.nodes) + [
        p for p in probe_nodes if p not in graph
    ]

    union_find = _UnionFind(nodes)

    for probe, grains in zip(probe_nodes, contacted):
        for grain in grains:
            union_find.union(probe, grain)

    groups: dict[object, list[object]] = {}

    for node in nodes:
        root = union_find.find(node)
        groups.setdefault(root, []).append(node)

    # Give each contracted group a stable label. If a group contains a probe,
    # use that probe label. Otherwise, use a tuple label for the grain group.
    root_to_label = {}

    for root, group_nodes in groups.items():
        probes_in_group = [
            p for p in probe_nodes if p in group_nodes
        ]

        if probes_in_group:
            root_to_label[root] = probes_in_group[0]
        else:
            root_to_label[root] = ("grain_group", len(root_to_label))

    contracted = nx.Graph()

    for root, group_nodes in groups.items():
        label = root_to_label[root]
        contracted.add_node(label, original_nodes=group_nodes)

    # Aggregate conductances between contracted groups. This correctly
    # preserves parallel resistors that become parallel after contraction.
    conductances: dict[tuple[object, object], float] = {}

    for u, v, data in graph.edges(data=True):
        ru = union_find.find(u)
        rv = union_find.find(v)

        if ru == rv:
            # An edge inside a contracted ideal node is bypassed by the
            # ideal connection.
            continue

        resistance = float(
            data.get(
                "resistance",
                grain_boundary_resistance_ohm,
            )
        )

        if not np.isfinite(resistance) or resistance <= 0:
            raise ValueError(
                f"Edge ({u!r}, {v!r}) has invalid resistance."
            )

        lu = root_to_label[ru]
        lv = root_to_label[rv]

        # Canonical ordering makes the undirected edge key consistent.
        key = tuple(sorted((lu, lv), key=repr))

        conductances[key] = (
            conductances.get(key, 0.0) + 1.0 / resistance
        )

    for (u, v), conductance in conductances.items():
        contracted.add_edge(
            u,
            v,
            resistance=1.0 / conductance,
        )

    return contracted


def four_probe_resistance(
    morphology: Morphology,
    graph: nx.Graph,
    probe_spacing: float,
    current_amp: float = 1.0,
    grain_boundary_resistance_ohm: float = 1.0,
    probe_contact_radius: float = 0.0,
    grain_resistivity_ohm_m: float = 1e-5,
    film_thickness_m: float = 92e-9,
    effective_width_m: float = 200e-9,
    grain_edge_factor: float = 0.5,
    length_unit_to_m: float = 1e-6,
) -> dict:
    """Calculate four-probe transfer resistance.

    Probe order:
        P0: current source
        P1: inner voltage probe, left
        P2: inner voltage probe, right
        P3: current sink

    R4p = (V(P1) - V(P2)) / I

    Ideal probe contacts are implemented by node contraction, not tiny
    artificial resistors. Negative transfer resistance is reported for
    diagnosis rather than silently converted to a positive value.
    """
    if not np.isfinite(current_amp) or current_amp <= 0:
        raise ValueError("current_amp must be finite and positive.")

    if (
        not np.isfinite(grain_boundary_resistance_ohm)
        or grain_boundary_resistance_ohm <= 0
    ):
        raise ValueError(
            "grain_boundary_resistance_ohm must be finite and positive."
        )

    if probe_contact_radius < 0:
        raise ValueError("probe_contact_radius must be non-negative.")

    if morphology.n_grains == 0:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "reason": "No grains.",
        }

    if graph.number_of_nodes() == 0:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "reason": "Contact graph is empty.",
        }

    if (
        not np.isfinite(grain_edge_factor)
        or grain_edge_factor < 0
    ):
        raise ValueError(
            "grain_edge_factor must be finite and non-negative."
        )
    probe_xy = probe_positions(
        morphology.width,
        probe_spacing,
        y=morphology.height / 2,
    )

    probe_nodes = ["P0", "P1", "P2", "P3"]

    contacted = [
        _contact_grains(
            morphology,
            graph,
            x,
            y,
            probe_contact_radius,
        )
        for x, y in probe_xy
    ]

    missing = [
        probe_nodes[i]
        for i, grains in enumerate(contacted)
        if not grains
    ]

    if missing:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": (
                "No grains contacted by: " + ", ".join(missing)
            ),
        }
    # A single grain touched by two ideal probes shorts those probes.
    for i in range(4):
        for j in range(i + 1, 4):
            if set(contacted[i]).intersection(contacted[j]):
                return {
                    "percolates": False,
                    "resistance_ohm": float("inf"),
                    "probe_positions": probe_xy,
                    "reason": (
                        f"Ideal probes {probe_nodes[i]} and "
                        f"{probe_nodes[j]} contact the same grain. "
                        "The probe geometry must be resolved."
                    ),
                }
    # Estimate an effective intragrain resistance for every grain.
    grain_resistances = calculate_grain_resistances(
        morphology=morphology,
        grain_resistivity_ohm_m=grain_resistivity_ohm_m,
        film_thickness_m=film_thickness_m,
        effective_width_m=effective_width_m,
        length_unit_to_m=length_unit_to_m,
    )

    # Each contact edge receives:
    # R_edge = R_gb + alpha * (R_grain_i + R_grain_j)
    base_graph = nx.Graph()
    base_graph.add_nodes_from(graph.nodes(data=True))

    for u, v in graph.edges():
        if u not in grain_resistances or v not in grain_resistances:
            raise ValueError(
                f"Contact edge ({u!r}, {v!r}) does not correspond "
                "to two known grain indices."
            )

        r_grain_u = grain_resistances[u]
        r_grain_v = grain_resistances[v]

        edge_resistance = (
            grain_boundary_resistance_ohm
            + grain_edge_factor * (r_grain_u + r_grain_v)
        )

        base_graph.add_edge(
            u,
            v,
            resistance=float(edge_resistance),
            grain_boundary_resistance_ohm=(
                grain_boundary_resistance_ohm
            ),
            grain_i_resistance_ohm=r_grain_u,
            grain_j_resistance_ohm=r_grain_v,
        )

    # Merge ideal probe nodes with the grains they contact.
    network = _contract_ideal_probe_contacts(
        base_graph,
        probe_nodes,
        contacted,
        grain_boundary_resistance_ohm,
    )

    # An ideal probe can electrically short two probes if both contact the
    # same grain/group. This is a special case, not a normal finite reading.
    if nx.has_path(network, "P0", "P3") is False:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": "No conductive path between outer probes.",
        }

    if (
        "P1" not in network
        or "P2" not in network
        or not nx.has_path(network, "P0", "P1")
        or not nx.has_path(network, "P0", "P2")
        or not nx.has_path(network, "P0", "P3")
    ):
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": (
                "One or both voltage probes are outside the "
                "current-carrying connected component."
            ),
        }

    solved = solve_resistor_network(
        network,
        "P0",
        "P3",
        current_amp=current_amp,
    )

    if not solved["connected"]:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": solved["reason"],
        }

    voltage = solved["voltages"]

    if "P1" not in voltage or "P2" not in voltage:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "reason": "Inner-probe voltages were not solved.",
        }

    voltage_difference = voltage["P1"] - voltage["P2"]
    resistance = voltage_difference / current_amp

    return {
        "percolates": True,
        "resistance_ohm": float(resistance),
        "voltage_difference_v": float(voltage_difference),
        "probe_positions": probe_xy,
        "probe_contacts": {
            probe_nodes[i]: contacted[i]
            for i in range(4)
        },
        "voltages": {
            probe: float(voltage[probe])
            for probe in probe_nodes
        },
        "negative_transfer_resistance": bool(resistance < 0),
        "graph": network,
        "reason": (
            "Negative transfer resistance: inspect potential ordering "
            "and network geometry."
            if resistance < 0
            else None
        ),
    }

def four_probe_resistance_multiterminal(
    morphology: Morphology,
    graph: nx.Graph,
    probe_spacing: float,
    current_amp: float = 1.0,
    grain_boundary_resistance_ohm: float = 1.0,
    grain_resistivity_ohm_m: float = 1e-5,
    film_thickness_m: float = 92e-9,
    effective_width_m: float = 200e-9,
    probe_contact_radius: float = 0.0,
    length_unit_to_m: float = 1e-6,
    minimum_path_length_m: float = 1e-9,
) -> dict:
    """Four-probe resistance using a multi-terminal grain network.

    Each grain has:
      - one central electrical node;
      - one terminal per grain-grain contact;
      - probe terminals wherever a probe touches that grain.

    Each grain terminal is connected to its grain centre through an
    approximate intragrain resistor:
        R = rho * path_length / (film_thickness * effective_width)

    Each grain-grain contact is connected by R_gb.

    P0 injects current, P3 extracts current.
    P1 and P2 are floating voltage probes.

    This is a multi-terminal approximation, not a full continuum solution
    of current spreading inside a circular grain.
    """
    if not np.isfinite(current_amp) or current_amp <= 0:
        raise ValueError("current_amp must be finite and positive.")

    if (
        not np.isfinite(grain_boundary_resistance_ohm)
        or grain_boundary_resistance_ohm <= 0
    ):
        raise ValueError(
            "grain_boundary_resistance_ohm must be finite and positive."
        )

    if (
        not np.isfinite(grain_resistivity_ohm_m)
        or grain_resistivity_ohm_m <= 0
    ):
        raise ValueError(
            "grain_resistivity_ohm_m must be finite and positive."
        )

    if not np.isfinite(film_thickness_m) or film_thickness_m <= 0:
        raise ValueError("film_thickness_m must be finite and positive.")

    if not np.isfinite(effective_width_m) or effective_width_m <= 0:
        raise ValueError("effective_width_m must be finite and positive.")

    if not np.isfinite(length_unit_to_m) or length_unit_to_m <= 0:
        raise ValueError("length_unit_to_m must be finite and positive.")

    if minimum_path_length_m <= 0:
        raise ValueError("minimum_path_length_m must be positive.")

    if morphology.n_grains == 0 or graph.number_of_nodes() == 0:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "reason": "No grains or empty contact graph.",
        }

    probe_xy = probe_positions(
        morphology.width,
        probe_spacing,
        y=morphology.height / 2,
    )

    probe_nodes = ["P0", "P1", "P2", "P3"]

    contacted = [
        _contact_grains(
            morphology,
            graph,
            px,
            py,
            probe_contact_radius,
        )
        for px, py in probe_xy
    ]

    missing = [
        probe_nodes[i]
        for i, grain_ids in enumerate(contacted)
        if not grain_ids
    ]

    if missing:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": "No grains contacted by: " + ", ".join(missing),
        }

    # The original graph is a geometric contact graph. Build a separate
    # electrical graph so geometry and electrical topology remain distinct.
    electrical = nx.Graph()

    # One centre node for every grain.
    for grain in graph.nodes:
        electrical.add_node(("centre", grain))

    # Convert grain radius and centre-to-centre distances to metres.
    def grain_radius_m(grain):
        return float(morphology.radii[grain]) * length_unit_to_m

    def intragrain_resistance(path_length_m):
        path_length_m = max(
            float(path_length_m),
            minimum_path_length_m,
        )
        return (
            grain_resistivity_ohm_m
            * path_length_m
            / (film_thickness_m * effective_width_m)
        )

    # Create a separate terminal for each end of each grain-grain contact.
    for u, v in graph.edges:
        centre_u = morphology.centers[u]
        centre_v = morphology.centers[v]

        distance_m = (
            np.linalg.norm(centre_u - centre_v)
            * length_unit_to_m
        )

        ru = grain_radius_m(u)
        rv = grain_radius_m(v)

        # Approximate the interface position along the line joining centres.
        # At tangency these path lengths approach the grain radii.
        # For overlapping disks they shorten with centre separation.
        denominator = ru + rv

        if denominator <= 0:
            raise ValueError("Grain radii must be positive.")

        path_u = distance_m * ru / denominator
        path_v = distance_m * rv / denominator

        terminal_u = ("contact", u, v)
        terminal_v = ("contact", v, u)

        electrical.add_edge(
            ("centre", u),
            terminal_u,
            resistance=intragrain_resistance(path_u),
        )

        electrical.add_edge(
            ("centre", v),
            terminal_v,
            resistance=intragrain_resistance(path_v),
        )

        electrical.add_edge(
            terminal_u,
            terminal_v,
            resistance=grain_boundary_resistance_ohm,
        )

    # Add one probe terminal per physical probe. A probe touching several
    # grains connects to each through an intragrain resistance arm.
    for probe, (px, py), grain_ids in zip(
        probe_nodes,
        probe_xy,
        contacted,
    ):
        electrical.add_node(probe, is_probe=True)

        for grain in grain_ids:
            centre_xy = morphology.centers[grain]

            path_length_m = (
                np.hypot(
                    centre_xy[0] - px,
                    centre_xy[1] - py,
                )
                * length_unit_to_m
            )

            electrical.add_edge(
                probe,
                ("centre", grain),
                resistance=intragrain_resistance(path_length_m),
            )

    solved = solve_resistor_network(
        electrical,
        source="P0",
        sink="P3",
        current_amp=current_amp,
    )

    if not solved["connected"]:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": solved["reason"],
        }

    voltages = solved["voltages"]

    missing_voltages = [
        p for p in ("P1", "P2") if p not in voltages
    ]

    if missing_voltages:
        return {
            "percolates": False,
            "resistance_ohm": float("inf"),
            "probe_positions": probe_xy,
            "probe_contacts": {
                probe_nodes[i]: contacted[i]
                for i in range(4)
            },
            "reason": (
                "Voltage probes are not in the current-carrying component: "
                + ", ".join(missing_voltages)
            ),
        }

    voltage_difference = voltages["P1"] - voltages["P2"]
    resistance = voltage_difference / current_amp

    return {
        "percolates": True,
        "resistance_ohm": float(resistance),
        "voltage_difference_v": float(voltage_difference),
        "probe_positions": probe_xy,
        "probe_contacts": {
            probe_nodes[i]: contacted[i]
            for i in range(4)
        },
        "voltages": {
            p: float(voltages[p]) for p in probe_nodes
        },
        "negative_transfer_resistance": bool(resistance < 0),
        "graph": electrical,
        "reason": (
            "Negative transfer resistance; inspect the potential field."
            if resistance < 0
            else None
        ),
    }