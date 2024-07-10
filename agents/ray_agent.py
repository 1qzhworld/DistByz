# agents/ray_agent.py

import ray
import numpy as np
from typing import Any, List, Dict, Literal, Tuple

from networks.network import Network
from problems.abstract_problem import AbstractProblem
from problems.least_square_problem import LeastSquareProblem
from problems.logit_regression_problem import LogitRegressionProblem
from functions.quadratic_function import QuadraticFunction
from utils.utils import check_subset, generate_weights

AgentType = Literal['Reg', 'Byz']
# ProblemType = Literal['LeastSquareProblem', 'LogitRegressionProblem']
# ImplementedAlgorithm = Literal['DGDAlgorithm', 'BBAlgorithm']


@ray.remote
class RayAgent:
    """
    Ray agent with remote communication and computation. But no algorithm is implemented. Algorithm should be a thing to
    reorganize the functions in ray agent. And the basic functions, such as grad_desc, aggregation, broadcase, and so on
    should be implemented in the RayAgent. In the other words, only basic actions are included in RayAgent, and how to
    organize the actions are organized in the algorithm. This also helpful when writing the synchronized algorithms.
    NOTICE: when create agents, should provide the name in options:
    RayAgent(...).options(name=f"{agent_id}").remote().
    after create the instances, agent_setup() should be firstly executed.
    """
    def __init__(self, agent_id: int, agent_type: AgentType, _network: Network,
                 prob_settings: Dict[str, Any], param: np.ndarray = None) -> None:
        self.agent_id = agent_id
        self.agent_type = self._check_agent_type(agent_type)

        self.in_nbr_ids = _network.get_in_neighbors(agent_id)
        self.out_nbr_ids = _network.get_out_neighbors(agent_id)
        self.loc_wt = generate_weights(_network)[:, agent_id]
        self.cached_in_nbr_handles = {}
        self.cached_out_nbr_handles = {}

        self.inbox = {in_nbr_id: {"param": None, "iter": 0} for in_nbr_id in self.in_nbr_ids}
        self.outbox = {out_nbr_id: {"param": None, "iter": 0} for out_nbr_id in self.out_nbr_ids}

        self.param: np.ndarray = param if param else np.zeros(prob_settings['dimension'])
        self.prob = self._setup_problem(agent_id, prob_settings)


    def agent_setup(self) -> None:
        """This should be the first function called after __init__."""
        self._set_cached_handles()
        msgs = {out_nbr_id: {'param': self.param} for out_nbr_id in self.out_nbr_ids} if self.agent_type == 'Reg'\
            else {out_nbr_id: {'param': np.random.randn(*self.prob.dimension)} for out_nbr_id in self.out_nbr_ids}
        self._set_outbox(msgs)

    def _set_cached_handles(self) -> None:
        """Function is called in agent_setup(). The cached handles are used for send messages."""
        self.cached_in_nbr_handles = {in_nbr: ray.get_actor(f"{in_nbr}") for in_nbr in self.in_nbr_ids}
        self.cached_out_nbr_handles = {out_nbr: ray.get_actor(f"{out_nbr}") for out_nbr in self.out_nbr_ids}

    def recv_from_nbr(self, sender_id: int, msg: Any) -> Tuple[int, Dict[int, Any]]:
        """Passive receive, used in send_to_nbrs. Save received information to inbox."""
        self.inbox[sender_id] = msg
        return sender_id, msg

    def send_to_nbrs(self, used_out_nbr_ids: List[int], timeout=1.0) -> None:
        """Only send from outbox. Therefore, no message parameter is provided."""
        check_subset(used_out_nbr_ids, self.out_nbr_ids)
        futures = [self.cached_out_nbr_handles[out_nbr_id].recv_from_nbr.remote(sender_id=self.agent_id, msg=msg)
                   for out_nbr_id, msg in self.outbox.items()]
        _, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        if remaining_futures:
            raise TimeoutError("Timed out waiting for all neighbors to respond.")

    def grad_desc(self, lr: float) -> None:
        """lr should be provided, should be part of algorithm."""
        self.param = self.param - lr * self.loss.gd(self.param)
        # self.param = self.param - lr * neg_dir

    def broadcast(self) -> None:
        if self.agent_type == 'Reg':
            msgs = {out_nbr_id: {"param": self.param} for out_nbr_id in self.out_nbr_ids}
        elif self.agent_type == 'Byz':
            # raise ValueError("What is going on?")
            msgs = {out_nbr_id: {"param": np.random.rand(*self.param.shape)} for out_nbr_id in self.out_nbr_ids}
            # print(msgs)
        else:
            msgs = {out_nbr_id: {"param": "something wrong."} for out_nbr_id in self.out_nbr_ids}
        self._set_outbox(msgs)
        self.send_to_nbrs(used_out_nbr_ids=self.out_nbr_ids)

    def aggregation(self) -> None:
        tmp_param = np.sum([self.loc_wt[in_nbr_id] * self.inbox[in_nbr_id]['param'] for in_nbr_id in self.in_nbr_ids])
        self.param = self.loc_wt[self.agent_id] * self.param + tmp_param

    @staticmethod
    def _check_agent_type(agent_type: AgentType) -> str:
        if agent_type not in ['Reg', 'Byz']:
            raise ValueError(f"Invalid agent type: {agent_type}")
        return agent_type

    def _setup_problem(self, agent_id, prob_settings: Dict[str, Any]) -> None:
        if prob_settings['problem_type'] == 'LeastSquareProblem':
            self.prob = LeastSquareProblem(agent_id=agent_id, prob_settings=prob_settings)
            self.loss = QuadraticFunction(self.prob.A, self.prob.b, self.prob.c, self.param)
        elif prob_settings['problem_type'] == 'LogitRegressionProblem':
            self.prob = LogitRegressionProblem(agent_id=agent_id, prob_settings=prob_settings)
            # self.loss = TODO: not finished for logit regression problem.
        else:
            raise ValueError(f"Invalid problem type: {prob_settings['problem_type']}")

    def _set_outbox(self, msgs: Dict[int, Any]) -> None:
        self.outbox = msgs

    def get_prob(self) -> AbstractProblem:
        return self.prob

    def get_param(self) -> np.ndarray:
        return self.param

    def get_inbox(self) -> Dict[int, Any]:
        return self.inbox

    def get_outbox(self) -> Dict[int, Any]:
        return self.outbox

    def get_type(self) -> str:
        return self.agent_type

