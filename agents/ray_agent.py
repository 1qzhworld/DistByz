# agents/ray_agent.py
import warnings

import ray
import numpy as np
import math
from typing import Any, List, Dict, Literal, Tuple, Set

from networks.network import Network
from problems.abstract_problem import AbstractProblem
from problems.least_square_problem import LeastSquareProblem
from problems.logit_regression_problem import LogitRegressionProblem
from functions.quadratic_function import QuadraticFunction
from utils.utils import check_subset

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
    def __init__(
            self, agent_id: int, _network: Network, prob_settings: Dict[str, Any],
            alg_settings: Dict[str, Any], rst_settings: Dict[str, Any],
            param: np.ndarray = None, blk_size: int = None, wt_plan: str = "metropolis-Hastings"
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = self._check_agent_type(_network.agent_types[agent_id])

        self._network = _network
        self.byz_size = _network.byz_size
        self.in_nbr_ids = _network.get_in_neighbors(agent_id)
        self.out_nbr_ids = _network.get_out_neighbors(agent_id)

        self.cached_in_nbr_handles = {}
        self.cached_out_nbr_handles = {}

        self.inbox = {in_nbr_id: {"param": None, "iter": 0} for in_nbr_id in self.in_nbr_ids}
        self.outbox = {out_nbr_id: {"param": None, "iter": 0} for out_nbr_id in self.out_nbr_ids}

        self.param: np.ndarray = param if isinstance(param, np.ndarray) else np.zeros(prob_settings['dimension'])
        self._setup_problem(agent_id, prob_settings)
        self.M = prob_settings['A']['HessUpperBd']
        self.m = prob_settings['A']['HessLowerBd']

        self.alg_settings = alg_settings

        self.rst_settings = rst_settings

        self.blk_num, self.blk_size, self.blk_indexes = self.setup_blk(blk_size)
        if alg_settings['alg_name'] == 'BBAlgorithm':
            self.loc_blk_wts = {blk_num_id: np.zeros(self._network.net_size) for blk_num_id in range(self.blk_num)}
        elif alg_settings['alg_name'] == 'DGDAlgorithm':
            self.loc_wts = self.generate_weights(weights_plan=wt_plan)[:, agent_id]  # different from block weights.

    def agent_setup(self) -> None:
        """This should be the first function called after __init__."""
        self._set_cached_handles()
        msgs = {out_nbr_id: {'param': self.param} for out_nbr_id in self.out_nbr_ids} if self.agent_type == 'Reg'\
            else {out_nbr_id: {'param': np.random.randn(*self.prob.dimension)} for out_nbr_id in self.out_nbr_ids}
        self._set_outbox(msgs)

    def setup_blk(self, blk_size: int) -> Tuple[int, int, Dict[int, List[int]]]:
        """
        blk_size: size of block (the last block may be smaller)
        blk_num: blk_size*blk_num >= blk_dim
        return blk_num, blk_size, blk_idxes
        """
        print(f"block size is: {blk_size}.")
        blk_size = blk_size if isinstance(blk_size, int) else self.prob.dimension
        if 2 * blk_size * self.byz_size + 1 > len(self.in_nbr_ids):  # check r-robust condition.
            warnings.warn(f"The block size {blk_size} is too large for {self.byz_size} byzantine agents.")
        return self._divide_blks(blk_size)

    def _divide_blks(self, blk_size: int) -> Tuple[int, int, Dict[int, List[int]]]:
        indices = np.arange(self.prob.dimension)
        np.random.shuffle(indices)
        blk_num = math.ceil(self.prob.dimension / blk_size)
        blk_idxes = {}
        for i in range(blk_num):
            if i == blk_num - 1:  # for the last block (maybe smaller than other blocks)
                blk_idx = indices[i * blk_size:]
            else:  # For other blocks, take blk_dim indices
                blk_idx = indices[i * blk_size:(i + 1) * blk_size]
            blk_idxes[i] = blk_idx.tolist()  # Convert to list for easier use
        return blk_num, blk_size, blk_idxes

    def local_filter_scalar(self, dim: int) -> List[int]:
        """
        Filter out the largest and smallest extreme values based on a specified dimension.
        self._network.byz_size: int
        return the agent_ids that are in the middle.
        """
        # Retrieve the parameters from the inbox and include the current agent's parameters.
        id_params: Dict[int, np.ndarray] = self.inbox2params()
        # Sort the agents based on the specified dimension in their parameters.
        sorted_agents = sorted(id_params.items(), key=lambda item: item[1][dim])
        print(sorted_agents)
        sorted_ids = [agent_id for agent_id, _ in sorted_agents]

        byz_size = self.byz_size
        loc_index = sorted_ids.index(self.agent_id)
        # print(loc_index)
        # # Select the IDs to remove the highest and lowest values, taking into account the byz_size.
        # rm_high_ids = sorted_ids[max(loc_index + 1, len(sorted_ids) - byz_size):]
        # rm_low_ids = sorted_ids[:min(loc_index, byz_size)]
        # # Get all agent IDs and remove the identified high and low extremes.
        # sel_ids = [agent_id for agent_id in sorted_ids
        #            if (agent_id not in rm_high_ids) and (agent_id not in rm_low_ids)]
        sel_ids = sorted_ids[min(loc_index, byz_size):max(loc_index + 1, len(sorted_ids) - byz_size)]
        return sel_ids  # id of selected agents among dimension dim

    def local_filter_block(self, block: List[int]) -> List[int]:
        """return: id of selected agents among the block (dimensions), intersection of dimension results."""
        if not block:  # If block is empty, return an empty list
            return []
        sel_ids: Set[int] = set(self.local_filter_scalar(block[0]))  # Initialize the set with the selected IDs.
        for dim in block[1:]:  # Intersect with selected IDs from each subsequent dimension
            sel_ids &= set(self.local_filter_scalar(dim))
        return list(sel_ids)

    def load_loc_params(self) -> Dict[int, np.ndarray]:
        """ Convert the parameters in inbox (self included)."""
        id_params = {agent_id: msg['param'] for agent_id, msg in self.get_inbox().items()}
        id_params[self.agent_id] = self.param
        return id_params

    def _set_cached_handles(self) -> None:
        """Function is called in agent_setup(). The cached handles are used for send messages."""
        self.cached_in_nbr_handles = {in_nbr: ray.get_actor(f"{in_nbr}") for in_nbr in self.in_nbr_ids}
        self.cached_out_nbr_handles = {out_nbr: ray.get_actor(f"{out_nbr}") for out_nbr in self.out_nbr_ids}

    def send_to_nbrs(self, used_out_nbr_ids: List[int], timeout=1.0) -> None:
        """Only send from outbox. Therefore, no message parameter is provided."""
        check_subset(used_out_nbr_ids, self.out_nbr_ids)
        futures = [self.cached_out_nbr_handles[out_nbr_id].recv_from_nbr.remote(sender_id=self.agent_id, msg=msg)
                   for out_nbr_id, msg in self.outbox.items()]
        _, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        # TODO: for asynchronize the wait can be removed.
        if remaining_futures:
            raise TimeoutError("Timed out waiting for all neighbors to respond.")

    def grad_desc(self, lr: float) -> None:
        """lr should be provided, should be part of algorithm."""
        self.param = self.param - lr * self.loss.gd(self.param)
        # self.param = self.param - lr * neg_dir
        # print(f"\rAgent {self.agent_id} has {self.param}\r")

    def recv_from_nbr(self, sender_id: int, msg: Any) -> Tuple[int, Dict[int, Any]]:
        """Passive receive, used in send_to_nbrs. Save received information to inbox."""
        self.inbox[sender_id] = msg
        return sender_id, msg

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
        tmp_param = np.sum([self.loc_wts[in_nbr_id] * self.inbox[in_nbr_id]['param']
                            for in_nbr_id in self.in_nbr_ids], axis=0)
        self.param = self.loc_wts[self.agent_id] * self.param + tmp_param

    def generate_weights(self, weights_plan: str = 'metropolis-Hastings') -> np.ndarray:  # changed default weights_plan for BBAlg
        # TODO: The generation plan is unreasonable using the network information,
        #  improvement on the algorithm can be a direction for research.
        if weights_plan == 'metropolis-Hastings':
            return self._generate_metropolis_hastings_weights()
        else:
            raise ValueError(f"Unknown weights plan: {weights_plan}")

    def _generate_metropolis_hastings_weights(self) -> np.ndarray:
        """
            weight w_{ij} for x_{i}^{t+1} = \\sum_{j \\in N(i)} w_{ij} x_{j}^{t}.
            :return: the weight above
        """
        net_size = self._network.net_size
        out_degrees = np.array([len(self._network.get_out_neighbors(agent_id)) for agent_id in range(net_size)])
        weights = np.zeros((net_size, net_size))

        for agent_id in range(net_size):
            in_nbrs = self._network.get_in_neighbors(agent_id)
            for in_nbr in in_nbrs:  # loc_wt for in-neighbors (but set the weight by out-degree)
                weights[agent_id, in_nbr] = 1 / (np.maximum(out_degrees[agent_id], out_degrees[in_nbr]) + 1)
        for agent_id in range(net_size):
            weights[agent_id, agent_id] = 1 - np.sum(weights[:, agent_id])
        return weights

    def generate_block_weights(self, weights_plan: str = 'block_byz') -> Dict[int, np.ndarray]:
        if weights_plan == 'block_byz':
            return self._generate_block_byz_weights()
        else:
            raise ValueError(f"Unknown __block__ weights plan: {weights_plan}.")

    def _generate_block_byz_weights(self) -> Dict[int, np.ndarray]:
        block_wts = {}
        for blk_num_id, blk_index in self.blk_indexes.items():  # blk_id (in range(blk_num)) which block, blk_i
            sel_ids = self.local_filter_block(blk_index)
            check_subset(sel_ids, self._network.get_in_neighbors(self.agent_id))
            block_wts[blk_num_id] = self._generate_block_byz_weight(sel_ids)  # just a row vector, not matrix.
        return block_wts

    def _generate_block_byz_weight(self, sel_ids: List[int]) -> np.ndarray:
        net_size = self._network.net_size
        wx = 1/net_size  # lower bound of weight for x variable.
        # The weight can be changed to other values such as 1/(self._network.get_out_neighbors(self.agent_id)).
        reg_size = net_size - self.byz_size
        gamma = 0.5*(1+1/(1-self.m/(self.M+self.m)))  # self.M+self.m should be self.M, but use the sum for safe.
        wd = np.max((reg_size-1-gamma)/(reg_size-2), wx)  # diagonal of weight should be positive.
        check_subset([self.agent_id], sel_ids)

        w_others = (1-wd) / (len(sel_ids)-1)  # sum of weights without agent_id itself.
        loc_wt_row = np.zeros(net_size)
        loc_wt_row[sel_ids] = w_others
        loc_wt_row[self.agent_id] = wd  # replace the `diagonal` one with its value.
        return loc_wt_row

    # def one_blk_agg(self, blk_num_id: int, blk_index: List[int]) -> Dict[Tuple[int], np.ndarray]:
    #     """
    #     Args:
    #         blk_num_id: the blk_num_id-th block (in range(self.blk_num))
    #         blk_index: the index of the corresponding blk_num_id (self.blk_indexes[blk_num_id]).
    #     Returns:
    #         One block aggregation with blockwise weight matrix and blockwise aggregation.
    #         return the aggregated vector (not full vector) whose length is the block dimension.
    #         Its key in tuple is the indexes (block_index) of the vector in the full vector.
    #     """
    #     id_params = self.loc_params()
    #     id_param_blk_idex = {agent_id: id_params[agent_id] for agent_id in blk_index}
    #         self.loc_wts[blk_id]
    #         # TODO add vector dimensions.
    #         blk_agg = 1  # TODO
    #     full_agg = np.zeros(self.prob.dimension)  # TODO
    #     for agent_id in blk_indexes:
    #         full_agg += self.loc_wts * self.get_inbox()[agent_id]['param']
    #     return full_agg

    def block_byz_aggregation(self) -> np.ndarray:
        """
        block aggregation,
        return the full vector with blockwise aggregation through blockwise weights.
        """
        tmp_param = np.zeros(self.prob.dimension)
        loc_params = self.load_loc_params()  # self included.
        for dim in range(self.prob.dimension):
            for blk_num_id in range(self.blk_num):
                for agent_j in self.blk_indexes[blk_num_id]:
                    tmp_param[dim] += self.loc_blk_wts[blk_num_id][self.agent_id, agent_j]*loc_params[agent_j][dim]
        return tmp_param


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

    def get_byz_size(self):
        return self.byz_size

    def get_blk_info(self) -> Tuple[int, int, Dict[int, List[int]]]:
        return self.blk_num, self.blk_size, self.blk_indexes

    def get_loc_blk_wts(self):
        return self.loc_blk_wts
