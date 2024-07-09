# algorithms/abstract_algorithm.py

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Literal
import numpy as np
from copy import deepcopy

from functions.quadratic_function import QuadraticFunction
from problems.abstract_problem import AbstractProblem
from problems.least_square_problem import LeastSquareProblem
from networks.network import Network
from communicators.ray_communicator import RayCommunicator

AgentType = Literal['Reg', 'Byz']


class AbstractAlgorithm(ABC):
    def __init__(self, agent_id: int, agent_type: AgentType, network: Network,
                 problem: AbstractProblem, comm: RayCommunicator,
                 alg_settings: Dict[str, Any], result_setting: Dict[str, Any], agent_param_init: np.ndarray = None):
        self.agent_id = agent_id
        self.agent_type = agent_type

        self.in_nbr_ids = network.get_in_neighbors(agent_id)
        self.out_nbr_ids = network.get_out_neighbors(agent_id)
        self.network = network

        self.comm = comm

        self.prob = problem

        self.agent_param = agent_param_init if agent_param_init else np.zeros(self.prob.dimension)
        self.in_nbr_msgs = {in_nbr_id: {"param": deepcopy(self.agent_param),
                                        "iter": None} for in_nbr_id in self.in_nbr_ids}

        self.alg_settings = alg_settings
        self.end_iter = self.alg_settings['end_iter']
        self.iter = 0

        self.loc_wt = self._generate_weights(network)[:, agent_id]  # column
        self.stepsize = self.alg_settings['stepsize']
        self.lr = None  # dependent on the stepsize, determined in self.run().

        if isinstance(self.prob, LeastSquareProblem):
            self.loss = QuadraticFunction(self.prob.A, self.prob.b, self.prob.c, self.agent_param)
        else:
            raise NotImplementedError(f'Problem {type(self.prob).__name__} is not implemented.')

        self.res_setting = result_setting
        self.res_record = self.res_setting['record']
        self.res_save = self.res_setting['save']
        self.res_func_eval = self.res_setting['func_eval']
        self.res_dist_to_opt = self.res_setting['dist_lc_opt']
        self.res_load = self.res_setting['load']

        self.func_eval = np.zeros(self.end_iter)
        if self.res_dist_to_opt:
            self.lc_opt = self.loss.closed_form_minimizer()
            self.dist2opt = np.zeros(self.end_iter)
        else:
            self.lc_opt = None
            self.dist2opt = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def iter_run(self):
        pass

    @classmethod
    def _generate_weights(cls, network: Network, weights_plan: str = 'metropolis-Hastings') -> np.ndarray:
        if weights_plan == 'metropolis-Hastings':
            return cls._get_metropolis_hastings_weights(network)
        else:
            raise NotImplementedError(f'Weights plan {weights_plan} is not implemented.')

    @staticmethod
    def _get_metropolis_hastings_weights(network: Network) -> np.ndarray:
        """
        weight w_{ij} for x_{i}^{t+1} = \\sum_{j \\in N(i)} w_{ij} x_{j}^{t}.
        :param network: networkx object, directed or Undirected graph.
        :return:
        """
        net_size = network.net_size
        out_degrees = np.array([len(network.get_out_neighbors(agent_id)) for agent_id in range(net_size)])
        weights = np.zeros((net_size, net_size))

        for agent_id in range(net_size):
            in_nbrs = network.get_in_neighbors(agent_id)
            for in_nbr in in_nbrs:  # loc_wt for in-neighbors (but set the weight by out-degree)
                weights[agent_id, in_nbr] = 1 / (np.maximum(out_degrees[agent_id], out_degrees[in_nbr]) + 1)
        for agent_id in range(net_size):
            weights[agent_id, agent_id] = 1 - np.sum(weights[:, agent_id])

        return weights
