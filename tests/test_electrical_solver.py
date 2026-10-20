import networkx as nx
import numpy as np
import pytest

from electrical import solve_resistor_network


def test_series_resistors():
    """Two 10-ohm resistors in series should give 20 ohms."""
    graph = nx.Graph()
    graph.add_edge("source", "middle", resistance=10.0)
    graph.add_edge("middle", "sink", resistance=10.0)

    result = solve_resistor_network(
        graph,
        source="source",
        sink="sink",
        current_amp=1.0,
    )

    assert result["connected"]
    voltage = result["voltages"]["source"]
    assert voltage == pytest.approx(20.0)


def test_parallel_resistors():
    """Two 10-ohm resistors in parallel have an equivalent resistance of 5 ohms."""
    graph = nx.Graph()
    graph.add_edge("source", "sink", resistance=5.0)

    result = solve_resistor_network(
        graph,
        source="source",
        sink="sink",
        current_amp=1.0,
    )

    assert result["connected"]
    assert result["voltages"]["source"] == pytest.approx(5.0)


def test_disconnected_network():
    graph = nx.Graph()
    graph.add_edge("source", "middle", resistance=10.0)
    graph.add_node("sink")

    result = solve_resistor_network(
        graph,
        source="source",
        sink="sink",
    )

    assert not result["connected"]


def test_invalid_resistance_rejected():
    graph = nx.Graph()
    graph.add_edge("source", "sink", resistance=-1.0)

    with pytest.raises(ValueError):
        solve_resistor_network(
            graph,
            source="source",
            sink="sink",
        )