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
import sys
import os
# from datetime import datetime

from functions.quadratic_function import QuadraticFunction
from networks.network import Network
from problems.abstract_problem import AbstractProblem
from agents.ray_agent import RayAgent
from utils.utils import parse_args, centralized_solver, save_config, generate_log_path
from configs.configs import project_root

sys.path.append(project_root)  # add root to sys.path
# os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用Ray的日志去重


class AbstractAlgorithm(ABC):
    def __init__(self, args, common_network: Network, common_problems: AbstractProblem) -> None:
        self.args = args
        # self.network = Network(net_settings=args.net_settings)
        self.network = common_network
        print(self.network.agent_types)

        self.common_problems = common_problems

        np.random.seed(self.network.net_size)  # setup random seed for algorithms

        self.end_iter = args.alg_settings['end_iter']
        self.stepsize = args.alg_settings['stepsize']
        self.cur_iter = 0
        self.lr = self.gene_stepsize()
        self.timeout_base = args.alg_settings['timeout_base']

        self.agents = {}
        self.param_init = args.alg_settings.get('param_init', 0) * np.ones(args.prob_settings['dimension'])

        # self.prob_a_list = None  # set value in the self.centralized_solver (only regular agents)
        # self.prob_b_list = None  # set value in the self.centralized_solver

        self.verbose = args.alg_settings['verbose']
        self.rst_settings = args.result_settings

        self.logging_result_path = generate_log_path(
            config_path=self.args.all, timestamp=self.args.timestamp, ext='h5',
            filename_plan='alg_hist', alg_name=self.args.alg_settings['alg_name']
        )

    def setup_agents(self):
        """Setup Ray agents based on the provided network and arguments."""
        for agent_id in range(self.network.net_size):
            # noinspection PyUnresolvedReferences
            agent = RayAgent.options(name=f'{agent_id}').remote(
                agent_id=agent_id,
                _network=self.network,
                problem=self.common_problems[agent_id],
                alg_settings=self.args.alg_settings,
                rst_settings=self.args.result_settings,
                param=self.param_init
            )
            self.agents[agent_id] = agent

    def setup_communicators(self, setup_timeout):
        comm_setup_futures = [agent.agent_setup.remote() for agent in self.agents.values()]
        self.wait_for_all_futures(comm_setup_futures, setup_timeout, "setup of communicators")

    def run(self):
        print(f'\rSetup all agents for {self.args.alg_settings['alg_name']}.\r')
        self.setup_agents()
        # print(f"Generate the closed form global solution before update: {self.centralized_solver()}.")
        tmp_timeout = self.timeout_base * self.network.net_size  # timeout for synchronize comm, setup and others.
        setup_timeout = tmp_timeout if tmp_timeout > 20 else 20  # experimental value

        self.setup_communicators(setup_timeout=setup_timeout)
        # logging.info('Successfully setup all agents and their communicators.')
        # print('Successfully setup all agents and their communicators.')

        with tqdm(total=self.end_iter, desc="Training", unit="iter") as pbar:
            while self.cur_iter < self.end_iter:
                # iter_run is an abstract method, should be provided in the detailed algorithms.
                self.iter_run()

                self.print_runtime_info()
                pbar.set_postfix({
                    # 'Iteration': self.cur_iter,
                    'lr': f"{self.lr:.4f}"
                })
                pbar.update(1)
                self.cur_iter += 1
        logging.info('Shutdown ray.')

        # save the history parameters among agents
        if self.rst_settings['save_hist']:
            self.save_all_hist()
        if self.rst_settings['save_results']:
            self.save_config_settings()

        # load_all_hist_params = self.load_all_hist()
        # all_params = self.get_all_hist()
        # for k, v in load_all_hist_params.items():
        #     if np.all(v == all_params[k]):
        #         print("correct") \checked.
        #     else:
        #         print(f"Wrong! on agent_id: {k}")
        self.print_runtime_info()
        # ray.shutdown()

    @abstractmethod
    def iter_run(self):
        pass

    def print_runtime_info(self):  # for debug
        if self.verbose:
            if (self.cur_iter+1) % 50 == 0:
                print(f"Agent 0's outbox is {ray.get(self.agents[0].get_outbox.remote())}.")
                print(f"Agent 0's param is {ray.get(self.agents[0].get_param.remote())}.")

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
        futures = {agent_id: self.agents[agent_id].get_hist_param.remote() for agent_id in range(self.network.net_size)}
        tmp_timeout = self.timeout_base * self.network.net_size
        ready_get, _ = self.wait_for_all_futures(list(futures.values()), tmp_timeout, "get_hist_param")
        for agent_id, future in futures.items():
            if future in ready_get:
                all_hist_params[agent_id] = ray.get(future)
            else:
                raise RuntimeError("Failed to get hist params for agent {}".format(agent_id))

        return all_hist_params

    def save_all_hist(self):
        """filename_plan: default generate name using generate_filename().
        load_all_hist is provided in utils.utils."""
        all_params = self.get_all_hist()
        agent_types = self.network.agent_types
        with h5py.File(self.logging_result_path, 'w') as f:
            for agent_id, all_param in all_params.items():
                f.create_dataset(str(agent_id), data=all_param)
                # the i-th agent's k-th iteration will be load_all_params[i][k]
            f.create_dataset(
                "global_solution", data=centralized_solver(self.common_problems, self.network.agent_types)
            )
            f.create_dataset('config_file', data=os.path.basename(self.args.all))
            agent_ids = np.array(list(agent_types.keys()))
            agent_types_values = np.array(list(agent_types.values()), dtype=h5py.string_dtype())
            f.create_dataset("agent_ids", data=agent_ids)
            f.create_dataset("agent_types_values", data=agent_types_values)

    def save_config_settings(self):
        log_config_path = generate_log_path(config_path=self.args.all, timestamp=self.args.timestamp, ext='yml')
        save_config(self.args, log_config_path)


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
        blk_size = self.blk_size if isinstance(self.blk_size, int) else self.args.prob_settings['dimension']
        if 2 * blk_size * self.byz_size + 1 > self.min_in_degrees:  # check r-robust condition.
            warnings.warn(f"The block size {blk_size} is too large for {self.byz_size} byzantine agents.")
        return self._divide_blks(blk_size)

    def _divide_blks(self, blk_size: int) -> Dict[int, List[int]]:
        indices = np.arange(self.args.prob_settings['dimension'])
        np.random.shuffle(indices)
        blk_num = math.ceil(self.args.prob_settings['dimension'] / blk_size)
        blk_idxes = {}
        for i in range(blk_num):
            if i == blk_num - 1:  # for the last block (maybe smaller than other blocks)
                blk_idx = indices[i * blk_size:]
            else:  # For other blocks, take blk_dim indices
                blk_idx = indices[i * blk_size:(i + 1) * blk_size]
            blk_idxes[i] = blk_idx.tolist()  # Convert to list for easier use
        return blk_idxes


# if __name__ == "__main__":
#     abs_args = parse_args()
#     dgd_algorithm = AbstractAlgorithm(abs_args)
#     dgd_algorithm.run()
