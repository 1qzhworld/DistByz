# algorithms/bb_algorithm.py
import numpy as np

from networks.network import Network
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm


class BBAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        super(BBAlgorithm, self).__init__(args)

    def iter_run(self):
        setup_timeout = self.timeout_base * self.network.net_size
        bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

        agg_futures = [agent_handle.blks_byz_aggregation.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(agg_futures, setup_timeout, "blockwise byzantine aggregation")

        self.lr = self.gene_stepsize()

        # gradient descent update
        upd_futures = [agent_handle.grad_desc.remote(self.lr) for agent_handle in self.agents.values()]
        self.wait_for_all_futures(upd_futures, setup_timeout, "update")

    def generate_weights(self, weights_plan: str = 'metropolis-Hastings') -> np.ndarray:
        if weights_plan == 'metropolis-Hastings':
            return self._generate_metropolis_hastings_weights()
        if weights_plan == 'block_byz':
            return self._generate_block_byz_weights()
        else:
            raise ValueError(f"Unknown weights plan: {weights_plan}")

    def _generate_block_byz_weights(self) -> np.ndarray:
        pass


if __name__ == "__main__":
    bb_args = parse_args()
    bb_algorithm = BBAlgorithm(bb_args)
    bb_algorithm.run()
