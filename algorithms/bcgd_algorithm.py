# algorithms/bb_algorithm.py
import numpy as np
import os
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm, BlockInfoMixin
from agents.ray_agent import RayAgent


class BCGDAlgorithm(AbstractAlgorithm, BlockInfoMixin):
    def __init__(self, args):
        self.alg_name = self.__class__.__name__
        if args.alg_settings['alg_name'] != self.alg_name:
            raise ValueError(f"Setting for {args.alg_settings['alg_name']}, but here is {self.alg_name}.")
        AbstractAlgorithm.__init__(self, args)
        BlockInfoMixin.__init__(self, args, self.network)

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
        # if not self.verbose:
        #     for agent_id, agent in self.agents.items():
        #         print(f"after aggregation: weight on Agent {agent_id} is:\r")
        #         print(f"block weights (should be row stochastic)\r {ray.get(agent.get_loc_blk_wts.remote())}")
        #         print(ray.get(agent.get_loc_blk_wts.remote()))
        self.wait_for_all_futures(agg_futures, setup_timeout, "blockwise byzantine aggregation")

        # gradient descent update
        self.lr = self.gene_stepsize()
        upd_futures = [agent_handle.atc_grad_desc.remote(self.lr, self.cur_iter)
                       for agent_handle in self.agents.values()]
        self.wait_for_all_futures(upd_futures, setup_timeout, "update")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    bcgd_args = parse_args()
    bcgd_algorithm = BCGDAlgorithm(bcgd_args)
    bcgd_algorithm.run()
