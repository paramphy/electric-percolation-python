"""Analytical validation tests for the resistor-network solver."""
import networkx as nx
import pytest

from electrical import solve_resistor_network


def edge(graph, u, v, resistance):
    graph.add_edge(u, v, resistance=float(resistance))


def test_single_resistor():
    graph = nx.Graph()
    edge(graph, "source", "sink", 7.5)
    result = solve_resistor_network(graph, "source", "sink", current_amp=2.0)
    assert result["connected"]
    assert result["voltages"]["source"] == pytest.approx(15.0)
    assert (result["voltages"]["source"] - result["voltages"]["sink"]) / 2.0 == pytest.approx(7.5)


def test_two_resistors_in_series():
    graph = nx.Graph()
    edge(graph, "source", "mid", 3.0)
    edge(graph, "mid", "sink", 5.0)
    result = solve_resistor_network(graph, "source", "sink", current_amp=1.0)
    assert result["connected"]
    assert result["voltages"]["source"] == pytest.approx(8.0)
    assert result["voltages"]["mid"] == pytest.approx(5.0)


def test_two_resistors_in_parallel():
    graph = nx.Graph()
    # Two equal-resistance series branches in parallel have R_eq = 3 ohm.
    edge(graph, "source", "branch_a", 3.0)
    edge(graph, "branch_a", "sink", 3.0)
    edge(graph, "source", "branch_b", 3.0)
    edge(graph, "branch_b", "sink", 3.0)
    result = solve_resistor_network(graph, "source", "sink", current_amp=1.0)
    assert result["connected"]
    assert result["voltages"]["source"] == pytest.approx(3.0)
    assert result["voltages"]["branch_a"] == pytest.approx(1.5)
    assert result["voltages"]["branch_b"] == pytest.approx(1.5)


def test_disconnected_network_returns_no_finite_solution():
    graph = nx.Graph()
    edge(graph, "source", "a", 2.0)
    edge(graph, "b", "sink", 2.0)
    result = solve_resistor_network(graph, "source", "sink")
    assert not result["connected"]
    assert result["voltages"] == {}
    assert "No conductive path" in result["reason"]


def test_invalid_edge_resistance_is_rejected():
    graph = nx.Graph()
    edge(graph, "source", "sink", 0.0)
    with pytest.raises(ValueError, match="positive resistance"):
        solve_resistor_network(graph, "source", "sink")
