# algorithms/bb_algorithm.py
import numpy as np
import ray
from tqdm import tqdm
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm


class BBAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        super(BBAlgorithm, self).__init__(args)
        self.blk_size = args.prob_info['blk_size']

    def setup_agents(self):
        # setup with block-wise weight matrices.
        for agent_id in range(self.network.net_size):
            # noinspection PyUnresolvedReferences
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                prob_settings=self.args.prob_info
            )
            self.agents[agent_id] = agent

    # def run(self):
    #     print("Running BBAlgorithm... Starting ray.")
    #     ray.init(ignore_reinit_error=True)
    #     print("Setup agents.")
    #     self.setup_agents()
    #     print(f"Global solution for LSP is: {self.centralized_solver()}.")
    #     setup_timeout = max(20, self.timeout_base * self.network.net_size)
    #     self.setup_communicators(setup_timeout=setup_timeout)
    #     print('Successfully setup communicators.')
    #
    #     with tqdm(total=self.end_iter, desc="BB Algorithm", unit="iter") as pbar:
    #         while self.cur_iter < self.end_iter:
    #             self.iter_run()
    #
    #             self.print_runtime_info()
    #             pbar.set_postfix({
    #                 'Iteration': self.cur_iter,
    #                 'Learning Rate': f"{self.lr:.6f}"
    #             })
    #             pbar.update(1)
    #             self.cur_iter += 1
    #     ray.shutdown()

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


if __name__ == "__main__":
    bb_args = parse_args()
    bb_algorithm = BBAlgorithm(bb_args)
    bb_algorithm.run()
