# tests/test_network_save_and_load.py

import os
import unittest
import networkx as nx
from utils.utils import load_common_network, save_common_network
from networks.network import Network


class TestNetworkLoading(unittest.TestCase):

    def setUp(self):
        self.net_settings = {
            'net_size': 10,
            'byz_size': 2,
            'conn_req': True,
            'gene_plan': 'ER_graph',
            'directed_req': False,
            'ER_p': 0.5,
            'allow_gene_iter': 100,
            'allow_gene_time': 30,
        }

        self.shared_network = Network(self.net_settings)
        self.shared_network_filename = 'test_common_network_utils.joblib'
        save_common_network(self.shared_network, self.shared_network_filename)

    def tearDown(self):
        if os.path.exists(self.shared_network_filename):
            os.remove(self.shared_network_filename)

    def compare_networks(self, net1, net2):
        # Get the attributes of both networks
        net1_attrs = vars(net1)
        net2_attrs = vars(net2)

        # Compare the attributes, ignoring memory addresses and networkx graph objects
        for key in net1_attrs:
            if key in net2_attrs and key != 'network':  # 'network' is the networkx graph
                self.assertEqual(net1_attrs[key], net2_attrs[key], f"Attribute '{key}' differs")

        # Compare the networkx graph objects for isomorphism
        self.assertTrue(nx.is_isomorphic(net1.network, net2.network), "Network graphs are not isomorphic")

    def test_load_network(self):
        load_network = load_common_network(self.shared_network_filename)
        self.compare_networks(self.shared_network, load_network)

    def test_generated_load_network(self):
        net_settings_with_load = {
            'load': True,
            'load_filename': self.shared_network_filename,
            'net_size': 5,  # These settings should be ignored
            'byz_size': 3,  # These settings should be ignored
            'conn_req': True,
            'gene_plan': 'regular',
            'directed_req': False,
            'reg_deg': 3,
            'allow_gene_iter': 100,
            'allow_gene_time': 30,
        }
        generated_load_network = Network(net_settings_with_load)
        self.compare_networks(self.shared_network, generated_load_network)

    def test_generated_load_network2(self):
        net_settings_with_load2 = {
            'load': True,
            'load_filename': self.shared_network_filename,
        }
        generated_load_network2 = Network(net_settings_with_load2)
        self.compare_networks(self.shared_network, generated_load_network2)


if __name__ == '__main__':
    unittest.main()
