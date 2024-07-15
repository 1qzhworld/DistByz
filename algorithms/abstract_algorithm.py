# algorithms/abstract_algorithm.py

import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import ray
import math
import warnings
from typing import Dict, List
import h5py
from datetime import datetime

from functions.quadratic_function import QuadraticFunction
from networks.network import Network
from agents.ray_agent import RayAgent
from utils.utils import parse_args, generate_logging_path

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # get current path.
project_root = os.path.join(current_dir, '..')  # root directory
sys.path.append(project_root)  # add root to sys.path
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
        self.timeout_base = args.alg_settings['timeout_base']

        self.agents = {}

        self.verbose = args.alg_settings['verbose']
        self.rst_settings = args.result_settings

        self.logging_result_path = generate_logging_path(
            config_one_path=self.args.one, ext='h5', filename_plan='config_one'
        )

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
        prob_futures = [agent.get_prob.remote() for agent in self.agents.values()
                        if ray.get(agent.get_type.remote()) == 'Reg']
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
        print(f"Generate the closed form global solution before update: {self.centralized_solver()}.")
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

        # save the history parameters among agents
        if self.rst_settings['save_hist']:
            self.save_all_hist()
        # load_all_hist_params = self.load_all_hist()
        # all_params = self.get_all_hist()
        # for k, v in load_all_hist_params.items():
        #     if np.all(v == all_params[k]):
        #         print("correct") \checked.
        #     else:
        #         print(f"Wrong! on agent_id: {k}")
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

    def get_all_hist(self) -> Dict[int, np.ndarray]:
        """Save the history parameters among agents."""
        all_hist_params = {}
        for agent_id in range(self.network.net_size):
            all_hist_params[agent_id] = ray.get(self.agents[agent_id].get_hist_param.remote())
        return all_hist_params

    def save_all_hist(self):
        """filename_plan: default generate name using generate_filename()."""
        all_params = self.get_all_hist()

        with h5py.File(self.logging_result_path, 'w') as f:
            for agent_id, all_param in all_params.items():
                f.create_dataset(str(agent_id), data=all_param)

    def load_all_hist(self):
        with h5py.File(self.logging_result_path, 'r') as f:
            load_all_params = {int(agent_id): f[agent_id][()] for agent_id in f.keys()}
        return load_all_params


    def generate_filename(self, filename_plan: str = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # if not isinstance(filename_plan, str):
        #     raise TypeError("In any case, the filename plan should be string. Please check.")
        # if filename_plan is None:
        #     self.args.
        # else:  # set the plan as filename.
        #     return filename_plan
        pass


class BlockInfoMixin:
    """
    To generate a mixin class for the block algorithms, where the commonly used functions are defined here.
    blk_size: size of block (the last block may be smaller)
    blk_num: blk_size*blk_num >= blk_dim
    """
    def __init__(self, args, network):
        self.args = args
        self.blk_size = args.alg_settings['blk_size']

        self.byz_size = network.byz_size
        self.min_in_degrees = np.min([len(network.get_in_neighbors(agent_id))
                                      for agent_id in range(network.net_size)])

        self.blk_indexes = self.setup_blk_info()

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
    abs_args = parse_args()
    dgd_algorithm = AbstractAlgorithm(abs_args)
    dgd_algorithm.run()
