# algorithms/bb_algorithm.py
import numpy as np
import ray
import warnings
import math
from typing import List, Dict, Tuple
# from tqdm import tqdm
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm
from agents.ray_agent import RayAgent


class BBAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        self.alg_name = "BBAlgorithm"
        if args.alg_settings['alg_name'] != self.alg_name:
            raise ValueError(f"Setting for {args.alg_settings['alg_name']}, but here is BBAlgorithm.")
        super(BBAlgorithm, self).__init__(args)
        self.blk_size = self.args.alg_settings['blk_size']

        self.byz_size = self.network.byz_size
        self.min_in_degrees = np.min([len(self.network.get_in_neighbors(agent_id))
                                      for agent_id in range(self.network.net_size)])

        self.blk_indexes = self.setup_blk_info()

    def setup_agents(self):
        for agent_id in range(self.network.net_size):
            # noinspection PyUnresolvedReferences
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                prob_settings=self.args.prob_info,
                alg_settings=self.args.alg_settings,
                rst_settings=self.args.result_settings,
                param=np.random.rand(self.args.prob_info['dimension']),  # TODO: Delete this for aggregation test.
                blk_indexes=self.blk_indexes,
                wt_plan=self.args.alg_settings["wt_plan"]
            )
            self.agents[agent_id] = agent

    def iter_run(self):
        setup_timeout = self.timeout_base * self.network.net_size
        bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

        agg_futures = [agent_handle.block_byz_aggregation.remote() for agent_handle in self.agents.values()]
        if not self.verbose:
            for agent_id, agent in self.agents.items():
                print(f"after aggregation: weight on Agent {agent_id} is:\r")
                print(f"block weights (should be row stochastic)\r {ray.get(agent.get_loc_blk_wts.remote())}")
                # print(ray.get(agent.get_loc_blk_wts.remote()))
        self.wait_for_all_futures(agg_futures, setup_timeout, "blockwise byzantine aggregation")

        self.lr = self.gene_stepsize()

        # gradient descent update
        upd_futures = [agent_handle.grad_desc.remote(self.lr) for agent_handle in self.agents.values()]
        self.wait_for_all_futures(upd_futures, setup_timeout, "update")

    def setup_blk_info(self) -> Dict[int, List[int]]:
        """
        blk_size: size of block (the last block may be smaller)
        blk_num: blk_size*blk_num >= blk_dim
        return blk_num, blk_size, blk_idxes
        """
        print(f"block size is: {self.blk_size}.")
        blk_size = self.blk_size if isinstance(self.blk_size, int) else self.args.prob_info['dimension']
        if 2 * blk_size * self.byz_size + 1 > self.min_in_degrees:  # check r-robust condition.
            warnings.warn(f"The block size {blk_size} is too large for {self.byz_size} byzantine agents.")
        return self._divide_blks(blk_size)

    def _divide_blks(self, blk_size: int) -> Dict[int, List[int]]:
        indices = np.arange(self.args.prob_info['dimension'])
        np.random.shuffle(indices)
        blk_num = math.ceil(self.args.prob_info['dimension'] / blk_size)
        blk_idxes = {}
        for i in range(blk_num):
            if i == blk_num - 1:  # for the last block (maybe smaller than other blocks)
                blk_idx = indices[i * blk_size:]
            else:  # For other blocks, take blk_dim indices
                blk_idx = indices[i * blk_size:(i + 1) * blk_size]
            blk_idxes[i] = blk_idx.tolist()  # Convert to list for easier use
        return blk_idxes


if __name__ == "__main__":
    bb_args = parse_args()
    bb_algorithm = BBAlgorithm(bb_args)
    bb_algorithm.run()
