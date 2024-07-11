# networks/network.py

import time
import random
import networkx as nx
from typing import Dict, Literal, Union


GenePlan = Literal['ER_graph', 'aver_deg', 'reg_graph', 'geo_graph', 'load_fixed']


class Network:
    """
    The _network generation is designed with different strategies.
    """
    def __init__(self, net_size: int, gene_plan: GenePlan,
                 net_settings: Dict[str, Union[int, str, nx.Graph, nx.DiGraph]], byz_size: int = None) -> None:
        self.net_size = net_size
        self.byz_size = byz_size if byz_size else 0

        self.gene_plan = gene_plan
        self.net_settings = net_settings
        self.net_conn_req = net_settings['conn_req']
        self.directed_req = net_settings['directed_req']
        self.network = self._generate_connected_network() if self.net_conn_req else self._generate_network()
        self.agent_types = self.gene_agent_types()

    def _generate_connected_network(self) -> Union[nx.Graph, nx.DiGraph]:
        allow_generate_iteration: int = self.net_settings.get('allow_gene_iter', 1000)
        allow_generate_time: int = self.net_settings.get('allow_gene_time', 120)
        start_time = time.time()
        for _ in range(allow_generate_iteration):
            network = self._generate_network()
            if isinstance(network, nx.DiGraph):
                # print(f"Generated _network type: {type(_network)}")
                if nx.is_strongly_connected(network):
                    # print(f"Generated _network is strongly connected.")
                    return network
            elif isinstance(network, nx.Graph):
                if nx.is_connected(network):
                    return network
            if time.time() - start_time > allow_generate_time:
                break
        raise ValueError(f"Generated _network {allow_generate_iteration} iters and {time.time() - start_time} seconds, "
                         f"but cannot meet connectivity requirement.")

    def _generate_network(self) -> Union[nx.Graph, nx.DiGraph]:
        generators = {
            'ER_graph': self._generate_er_graph,
            'aver_deg': self._generate_aver_deg_graph,
            'reg_graph': self._generate_reg_graph,
            'geo_graph': self._generate_geo_graph,
            'load_fixed': self._load_fixed_graph,
        }
        if self.gene_plan in generators:
            return generators[self.gene_plan]()
        else:
            raise ValueError(f"Unknown gene_plan: {self.gene_plan}")

    def _generate_er_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        if 'ER_p' not in self.net_settings:
            raise ValueError("The parameter 'ER_p' is required in net_settings for ER (Erdos–Renyi) graph.")
        return nx.gnp_random_graph(self.net_size, self.net_settings['ER_p'], directed=self.directed_req)

    def _generate_aver_deg_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        if 'aver_deg' not in self.net_settings:
            raise ValueError("The parameter 'aver_deg' is required in net_settings for average degree graph.")
        aver_deg = self.net_settings['aver_deg']
        assert 0 < aver_deg < self.net_size, "aver_deg must be between 0 and net_size."
        return nx.gnp_random_graph(self.net_size, aver_deg / (self.net_size - 1), directed=self.directed_req)

    def _generate_reg_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        if 'reg_deg' not in self.net_settings:
            raise ValueError("The parameter 'reg_deg' is required in net_settings for regular graph.")
        reg_deg = self.net_settings['reg_deg']
        assert 0 < reg_deg < self.net_size, "reg_deg must be between 0 and net_size."
        assert (self.net_size * reg_deg) % 2 == 0, \
            "The product of net_size and reg_deg must be even for a regular graph."
        if self.directed_req:
            # Generate a random regular directed graph
            G = nx.DiGraph()
            nodes = list(range(self.net_size))
            for node in nodes:
                targets = set()
                while len(targets) < reg_deg:
                    target = node
                    while target == node or target in targets:
                        target = random.choice(nodes)
                    targets.add(target)
                for target in targets:
                    G.add_edge(node, target)
            return G
        else:
            return nx.random_regular_graph(reg_deg, self.net_size)

    def _generate_geo_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        if 'radius' not in self.net_settings:
            raise ValueError("The parameter 'radius' is required in net_settings for geometric graph.")
        return nx.DiGraph(nx.random_geometric_graph(self.net_size, self.net_settings['radius'])) if self.directed_req \
            else nx.random_geometric_graph(self.net_size, self.net_settings['radius'])

    def _load_fixed_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        if not (isinstance(self.net_settings['load_fixed'], nx.Graph) or
                isinstance(self.net_settings['load_fixed'], nx.DiGraph)):
            raise ValueError("The parameter 'load_fixed' must be a networkx Graph instance.")
        return self.net_settings['load_fixed']

    def gene_agent_types(self) -> Dict[int, str]:
        """
        return {agent_id: 'Byz' or 'Reg'}
        """
        reg_size = self.net_size - self.byz_size
        if reg_size > self.net_size:
            raise ValueError("reg_size cannot be greater than net_size")
        values = ['Reg'] * reg_size + ['Byz'] * self.byz_size
        random.shuffle(values)
        result_dict = {i: values[i] for i in range(self.net_size)}
        return result_dict

    def get_in_neighbors(self, agent_id: int) -> list[int]:  # self(agent_id) is not in return.
        if not self.network.has_node(agent_id):
            raise ValueError(f"Node {agent_id} does not exist in the _network.")
        if isinstance(self.network, nx.DiGraph):
            return list(self.network.predecessors(agent_id))
        else:
            return list(self.network.neighbors(agent_id))

    def get_out_neighbors(self, agent_id: int) -> list[int]:  # self(agent_id) is not in return.
        if not self.network.has_node(agent_id):
            raise ValueError(f"Node {agent_id} does not exist in the _network.")
        if isinstance(self.network, nx.DiGraph):
            return list(self.network.successors(agent_id))
        else:
            return list(self.network.neighbors(agent_id))
