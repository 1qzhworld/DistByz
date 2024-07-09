# agents/ray_agent_old.py

import ray
import numpy as np
from typing import Any, List, Dict, Literal

from networks.network import Network
from problems.least_square_problem import LeastSquareProblem
from problems.logit_regression_problem import LogitRegressionProblem
from problems.abstract_problem import AbstractProblem
from communicators.ray_communicator import RayCommunicator
from algorithms.dgd_algorithm import DGDAlgorithm
from algorithms.bb_algorithm import BBAlgorithm

AgentType = Literal['Reg', 'Byz']
ProblemType = Literal['LeastSquareProblem', 'LogitRegressionProblem']
ImplementedAlgorithm = Literal['DGDAlgorithm', 'BBAlgorithm']


class RayAgent:
    def __init__(self, agent_id: int, agent_type: AgentType, network: Network,
                 problem_type: ProblemType, dimension: int, prob_info: Dict[str, Any],
                 alg_name: ImplementedAlgorithm, alg_settings: Dict[str, Any], result_settings: Dict[str, Any],
                 param: np.ndarray = None):
        self.agent_id = agent_id
        self.agent_type = self._check_agent_type(agent_type)

        self._network = network
        self.in_nbr_ids = network.get_in_neighbors(agent_id)
        self.out_nbr_ids = network.get_out_neighbors(agent_id)
        self.comm = RayCommunicator.options(name=f"{agent_id}").remote(agent_id, self.in_nbr_ids, self.out_nbr_ids)

        self.prob = self._setup_problem(agent_id, problem_type, dimension, prob_info)
        self.param = param if param else np.zeros(self.prob.dimension)

        self.alg_name = alg_name
        self.alg_settings = alg_settings
        self.alg = self._setup_algorithm(agent_id, agent_type, network, self.prob, self.comm,
                                         alg_settings, result_settings)

        self.result_setting = result_settings

        self.inbox = []
        self.outbox = []

    def comm_setup(self):
        self.comm.set_cached_neighbors.remote()
        same_msg = {'param': self.param, 'iter': self.alg.iter}
        msgs = {out_nbr_id: same_msg for out_nbr_id in self.out_nbr_ids}
        self.comm.set_outbox.remote(msgs)
        # return self.comm  # for test

    def run(self):
        self.param = self.alg.run()

    def broadcast(self):
        return self.alg.broadcast()

    def get_comm(self):
        return self.comm

    def get_prob(self):
        return self.prob

    def get_alg(self):
        return self.alg

    def get_param(self):
        return self.param

    @staticmethod
    def _check_agent_type(agent_type: AgentType):
        if agent_type not in ['Reg', 'Byz']:
            raise ValueError(f"Invalid agent type: {agent_type}")
        return agent_type

    @staticmethod
    def _setup_problem(agent_id, problem_type: ProblemType, dimension: int, prob_info: Dict[str, Any]):
        if problem_type == 'LeastSquareProblem':
            return LeastSquareProblem(agent_id=agent_id, problem_type='LeastSquareProblem',
                                      dimension=dimension, prob_info=prob_info)
        elif problem_type == 'LogitRegressionProblem':
            return LogitRegressionProblem(agent_id=agent_id, problem_type='LogitRegressionProblem',
                                          dimension=dimension, prob_info=prob_info)
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    @staticmethod
    def _setup_algorithm(agent_id: int, agent_type: AgentType, network: Network,
                         prob: AbstractProblem, comm: RayCommunicator,
                         alg_settings: dict, result_settings: dict):
        alg_name = alg_settings['alg_name']
        if alg_name == "BBAlgorithm":
            return BBAlgorithm()  # TODO:implement this algorithm
        elif alg_name == "DGDAlgorithm":
            return DGDAlgorithm(agent_id, agent_type, network, prob, comm, alg_settings, result_settings)
        else:  # alg_name not in ImplementedAlgorithm:
            raise ValueError(f"Invalid algorithm name: {alg_name} (NOT IMPLEMENTED)")
