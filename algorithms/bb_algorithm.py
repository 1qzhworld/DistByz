# algorithms/bb_algorithm.py
import numpy as np
import ray
from tqdm import tqdm
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm
from agents.ray_agent import RayAgent

class BBAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        self.alg_name = "BBAlgorithm"
        super(BBAlgorithm, self).__init__(args)

    def setup_agents(self):
        # setup with block-wise weight matrices.
        for agent_id in range(self.network.net_size):
            # noinspection PyUnresolvedReferences
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                prob_settings=self.args.prob_info,
                param=np.random.rand(self.args.prob_info['dimension']),  # TODO: Delete this for aggregation test.
                blk_size=self.args.alg_settings['blk_size'],
                wt_plan=self.args.alg_settings["wt_plan"]
            )  # TODO: DO I need additional settings?
            self.agents[agent_id] = agent

    def iter_run(self):
        setup_timeout = self.timeout_base * self.network.net_size
        bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

        agg_futures = [agent_handle.block_byz_aggregation.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(agg_futures, setup_timeout, "blockwise byzantine aggregation")

        self.lr = self.gene_stepsize()

        # gradient descent update
        upd_futures = [agent_handle.grad_desc.remote(self.lr) for agent_handle in self.agents.values()]
        self.wait_for_all_futures(upd_futures, setup_timeout, "update")


if __name__ == "__main__":
    bb_args = parse_args()
    bb_algorithm = BBAlgorithm(bb_args)
    bb_algorithm.run()
