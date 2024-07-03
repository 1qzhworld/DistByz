import pytest
import networkx as nx
from networks.network import Network
from typing import Dict, Union

SettingsType = Dict[str, Union[int, float, str, bool, nx.Graph, nx.DiGraph]]


@pytest.fixture
def basic_settings() -> SettingsType:
    return {
        'conn_req': False,
        'directed_req': False,
    }


def test_er_graph(basic_settings):
    settings = basic_settings.copy()
    settings.update({'ER_p': 0.1})
    net = Network(net_size=10, gene_plan='ER_graph', net_settings=settings)
    assert isinstance(net.network, nx.Graph)


def test_aver_deg_graph(basic_settings):
    settings = basic_settings.copy()
    settings.update({'aver_deg': 2})
    net = Network(net_size=10, gene_plan='aver_deg', net_settings=settings)
    assert isinstance(net.network, nx.Graph)


def test_reg_graph(basic_settings):
    settings = basic_settings.copy()
    settings.update({'reg_deg': 2})
    net = Network(net_size=10, gene_plan='reg_graph', net_settings=settings)
    assert isinstance(net.network, nx.Graph)


def test_geo_graph(basic_settings):
    settings = basic_settings.copy()
    settings.update({'radius': 0.2})
    net = Network(net_size=10, gene_plan='geo_graph', net_settings=settings)
    assert isinstance(net.network, nx.Graph)


def test_load_fixed_graph(basic_settings):
    fixed_graph = nx.complete_graph(10)
    settings = basic_settings.copy()
    settings.update({'load_fixed': fixed_graph})
    net = Network(net_size=10, gene_plan='load_fixed', net_settings=settings)
    assert net.network is fixed_graph


def test_connected_network(basic_settings):
    settings = basic_settings.copy()
    settings.update({'ER_p': 0.5, 'conn_req': True})
    net = Network(net_size=10, gene_plan='ER_graph', net_settings=settings)
    assert nx.is_connected(net.network)


def test_directed_network(basic_settings):
    settings = basic_settings.copy()
    settings.update({'ER_p': 0.5, 'directed_req': True})
    net = Network(net_size=10, gene_plan='ER_graph', net_settings=settings)
    assert isinstance(net.network, nx.DiGraph)


def test_get_in_neighbors(basic_settings):
    settings = basic_settings.copy()
    settings.update({'ER_p': 0.1, 'directed_req': True})
    net = Network(net_size=10, gene_plan='ER_graph', net_settings=settings)
    node = list(net.network.nodes)[0]
    in_neighbors = net.get_in_neighbors(node)
    assert all(net.network.has_edge(neighbor, node) for neighbor in in_neighbors)


def test_get_out_neighbors(basic_settings):
    settings = basic_settings.copy()
    settings.update({'ER_p': 0.1, 'directed_req': True})
    net = Network(net_size=10, gene_plan='ER_graph', net_settings=settings)
    node = list(net.network.nodes)[0]
    out_neighbors = net.get_out_neighbors(node)
    assert all(net.network.has_edge(node, neighbor) for neighbor in out_neighbors)


def test_invalid_gene_plan(basic_settings):
    with pytest.raises(ValueError):
        Network(net_size=10, gene_plan='invalid_plan', net_settings=basic_settings)


def test_invalid_er_graph_params(basic_settings):
    with pytest.raises(ValueError):
        settings = basic_settings.copy()
        Network(net_size=10, gene_plan='ER_graph', net_settings=settings)


def test_invalid_aver_deg_params(basic_settings):
    with pytest.raises(ValueError):
        settings = basic_settings.copy()
        Network(net_size=10, gene_plan='aver_deg', net_settings=settings)


def test_invalid_reg_graph_params(basic_settings):
    with pytest.raises(ValueError):
        settings = basic_settings.copy()
        Network(net_size=10, gene_plan='reg_graph', net_settings=settings)


def test_invalid_geo_graph_params(basic_settings):
    with pytest.raises(ValueError):
        settings = basic_settings.copy()
        Network(net_size=10, gene_plan='geo_graph', net_settings=settings)


def test_invalid_load_fixed_graph_params(basic_settings):
    with pytest.raises(ValueError):
        settings = basic_settings.copy()
        settings.update({'load_fixed': 'not_a_graph'})
        Network(net_size=10, gene_plan='load_fixed', net_settings=settings)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
