# algorithms/bb_algorithm.py
import numpy as np
import ray
from tqdm import tqdm
from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm


class BBAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        super(BBAlgorithm, self).__init__(args)

    def setup_agents(self):


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

    def generate_weights(self, weights_plan: str = 'block_byz') -> np.ndarray:
        if weights_plan == 'metropolis-Hastings':
            return self._generate_metropolis_hastings_weights()
        if weights_plan == 'block_byz':
            return self._generate_block_byz_weights()
        else:
            raise ValueError(f"Unknown weights plan: {weights_plan}")

    def _generate_block_byz_weights(self) -> np.ndarray:
        raise NotImplementedError  # TODO!!!


if __name__ == "__main__":
    bb_args = parse_args()
    bb_algorithm = BBAlgorithm(bb_args)
    bb_algorithm.run()
