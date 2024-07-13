# algorithms/abstract_algorithm.py

import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import ray

from functions.quadratic_function import QuadraticFunction
from networks.network import Network
from agents.ray_agent import RayAgent
from utils.utils import parse_args

# os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用Ray的日志去重


class AbstractAlgorithm(ABC):
    def __init__(self, args):
        self.args = args
        self.network = Network(net_settings=args.net_settings)
        print(self.network.agent_types)

        self.end_iter = args.alg_settings['end_iter']
        self.stepsize = args.alg_settings['stepsize']
        self.cur_iter = 0
        self.lr = self.gene_stepsize()

        self.verbose = args.alg_settings['verbose']
        self.timeout_base = args.alg_settings['timeout_base']

        self.agents = {}

    def setup_agents(self):
        """Setup Ray agents based on the provided network and arguments."""
        for agent_id in range(self.network.net_size):
            # noinspection PyUnresolvedReferences
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                prob_settings=self.args.prob_info,
                alg_settings=self.args.alg_settings,
                rst_settings=self.args.result_settings
            )
            self.agents[agent_id] = agent

    def setup_communicators(self, setup_timeout):
        comm_setup_futures = [agent.agent_setup.remote() for agent in self.agents.values()]
        self.wait_for_all_futures(comm_setup_futures, setup_timeout, "setup of communicators")

    def centralized_solver(self) -> np.ndarray:
        """Compute the global centralized solution."""
        prob_futures = [agent.get_prob.remote() for agent in self.agents.values()]
        setup_timeout = self.timeout_base*self.network.net_size
        ready_probs, _ = self.wait_for_all_futures(prob_futures, setup_timeout, "get_prob")

        probs = [ray.get(ready_prob) for ready_prob in ready_probs]
        prob_a_list = [prob.A for prob in probs]
        sum_a = np.sum(prob_a_list, axis=0)
        probb_list = [prob.b for prob in probs]
        sum_b = np.sum(probb_list, axis=0)
        if self.verbose:
            print(f"A for agents are: {prob_a_list}")
            print(f"b for agents are {probb_list}")
        return QuadraticFunction(sum_a, sum_b).closed_form_minimizer()
        # return (-np.linalg.inv(0.5 * (sum_a.T + sum_a)) @ sum_b).flatten()

    def run(self):
        # logging.info("Start ray.")
        print("Starting ray.")
        ray.init(ignore_reinit_error=True)
        # logging.info('Start to setup agents.')
        print('Setup agents.')
        self.setup_agents()
        print(f"Global solution: {self.centralized_solver()}.")
        tmp_timeout = self.timeout_base * self.network.net_size  # timeout for synchronize comm, setup and others.
        setup_timeout = tmp_timeout if tmp_timeout > 20 else 20  # experimental value
        self.setup_communicators(setup_timeout=setup_timeout)
        # logging.info('Successfully setup all agents and their communicators.')
        print('Successfully setup all agents and their communicators.')

        with tqdm(total=self.end_iter, desc="Training Progress", unit="iter") as pbar:
            while self.cur_iter < self.end_iter:
                # iter_run is an abstract method, should be provided in the detailed algorithms.
                self.iter_run()

                self.print_runtime_info()
                pbar.set_postfix({
                    'Iteration': self.cur_iter,
                    'Learning Rate': f"{self.lr:.6f}"
                })
                pbar.update(1)
                self.cur_iter += 1
        logging.info('Shutdown ray.')
        ray.shutdown()

    @abstractmethod
    def iter_run(self):
        pass

    def print_runtime_info(self):
        if self.verbose:
            if self.cur_iter % 50 == 0:
                print(f"Agent 0's outbox is {ray.get(self.agents[0].get_outbox.remote())}.")
                print(f"Agent 0's paramt is {ray.get(self.agents[0].get_param.remote())}.")

    def gene_stepsize(self):
        if isinstance(self.stepsize, float):
            lr = self.stepsize
        elif self.stepsize == 'diminishing_stepsize':
            lr = self.diminishing_stepsize(self.cur_iter)
        else:
            raise ValueError('Stepsize in algorithm setting is incorrect.')
        return lr

    @staticmethod
    def wait_for_all_futures(futures, timeout, action_name):
        """Wait for the provided futures to complete or timeout."""
        ready, remaining = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        if remaining:
            raise TimeoutError(f"Timeout reached, {len(remaining)} {action_name} did not respond in {timeout}s.")
            # logging.warning(f"Timeout reached, {len(remaining)} {action_name} did not respond in time {timeout}s.")
        return ready, remaining

    @staticmethod
    def diminishing_stepsize(cur_iter: int) -> float:
        """Compute the diminishing stepsize."""
        return 1 / (cur_iter + 1)


if __name__ == "__main__":
    dgd_args = parse_args()
    dgd_algorithm = AbstractAlgorithm(dgd_args)
    dgd_algorithm.run()
