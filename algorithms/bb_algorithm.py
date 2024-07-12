# algorithms/bb_algorithm.py

import os
import logging

import numpy as np
import ray
from networks.network import Network
from agents.ray_agent import RayAgent
from utils.utils import parse_args, generate_weights

# os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用Ray的日志去重


class BBAlgorithm:
    def __init__(self, args):
        self.args = args
        self.network = Network(net_size=args.net_size, gene_plan=args.gene_plan, net_settings=args.net_settings)
        print(self.network.agent_types)
        self.stepsize = args.alg_settings['stepsize']
        self.end_iter = args.alg_settings['end_iter']
        self.alg_name = "DGDAlgorithm"
        if args.alg_settings['alg_name'] != self.alg_name:
            raise ValueError(f"Setting for {args.alg_settings['alg_name']}, but here is DGDAlgorithm.")
        self.verbose = args.alg_settings['verbose']
        self.cur_iter = 0
        self.timeout_base = args.alg_settings['timeout_base']
        self.agents = {}

        # additional initialization for the weight setting requirements.
        self.L_smooth = self.args.prob_info['A']['HessUpperBd']
        self.mu_strongly = self.args.prob_info['A']['HessLowerBd']
        self.gamma = (1+(1/(1-self.mu_strongly/self.L_smooth)))/2

        # Byzantine and regular information
        self.byz_size = self.network.byz_size
        self.reg_size = self.network.net_size - self.byz_size
        self.net_size = self.network.net_size

    def setup_agents(self):
        """Setup Ray agents based on the provided network and arguments."""
        for agent_id in range(self.network.net_size):
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                prob_settings=self.args.prob_info
            )
            self.agents[agent_id] = agent

    def setup_comms(self, setup_timeout):
        comm_setup_futures = [agent.agent_setup.remote() for agent in self.agents.values()]
        self.wait_for_all_futures(comm_setup_futures, setup_timeout, "setup of communicators")

    @staticmethod
    def wait_for_all_futures(futures, timeout, action_name):
        """Wait for the provided futures to complete or timeout."""
        ready, remaining = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        if remaining:
            raise TimeoutError(f"Timeout reached, {len(remaining)} {action_name} did not respond in time {timeout}s.")
            # logging.warning(f"Timeout reached, {len(remaining)} {action_name} did not respond in time {timeout}s.")
        return ready, remaining

    def global_centralized_sol(self) -> np.ndarray:
        """Compute the global centralized solution."""
        prob_futures = [agent.get_prob.remote() for agent in self.agents.values()]
        ready_probs, _ = self.wait_for_all_futures(prob_futures, 10, "get_prob")
        probs = [ray.get(ready_prob) for ready_prob in ready_probs]
        probA_list = [prob.A for prob in probs]
        print(f"A for agents are: {probA_list}")
        sum_a = np.sum(probA_list, axis=0)
        probb_list = [prob.b for prob in probs]
        sum_b = np.sum(probb_list, axis=0)
        print(f"b for agents are {probb_list}")
        return (-np.linalg.inv(0.5 * (sum_a.T + sum_a)) @ sum_b).flatten()

    @staticmethod
    def diminishing_stepsize(cur_iter: int) -> float:
        """Compute the diminishing stepsize."""
        return 1 / (cur_iter + 1)

    def run(self):
        logging.info("Start ray.")
        print("Starting ray.")
        ray.init(ignore_reinit_error=True)

        logging.info('Start to setup agents.')
        print('Setup agents.')
        self.setup_agents()

        print(f"Global solution: {self.global_centralized_sol()}.")
        tmp_timeout = self.timeout_base * self.network.net_size  # timeout for synchronize comm, setup and others.
        setup_timeout = tmp_timeout if tmp_timeout > 20 else 20  # experimental value
        self.setup_comms(setup_timeout=setup_timeout)

        logging.info('Successfully setup all agents.')
        print('Successfully setup all agents.')


        while self.cur_iter < self.end_iter:
            # print(f'----Current iteration: {self.cur_iter}.----')
            bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
            self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

            # aggregation
            agg_futures = [agent_handle.aggregation.remote() for agent_handle in self.agents.values()]
            self.wait_for_all_futures(agg_futures, setup_timeout, "aggregation")

            if isinstance(self.stepsize, float):
                lr = self.stepsize
            elif self.stepsize == 'diminishing_stepsize':
                lr = self.diminishing_stepsize(self.cur_iter)
            else:
                raise ValueError('Stepsize in algorithm setting is incorrect.')

            # gradient descent update
            upd_futures = [agent_handle.grad_desc.remote(lr) for agent_handle in self.agents.values()]
            self.wait_for_all_futures(upd_futures, setup_timeout, "update")

            if self.verbose:
                if self.cur_iter % 50 == 0:
                    print(f"Agent 0's outbox is {ray.get(self.agents[0].get_outbox.remote())}.")
                    print(f"Agent 0's paramt is {ray.get(self.agents[0].get_param.remote())}.")
            self.cur_iter += 1

        logging.info('Shutdown ray.')
        ray.shutdown()


def main():
    args = parse_args()
    dgd_algorithm = BBAlgorithm(args)
    dgd_algorithm.run()


if __name__ == "__main__":
    main()
