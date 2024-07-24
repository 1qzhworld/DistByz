# agents/ray_agent.py

import ray
import numpy as np
from typing import Any, List, Dict, Literal, Tuple, Set
import os

from networks.network import Network
from problems.abstract_problem import AbstractProblem
from problems.least_square_problem import LeastSquareProblem
from problems.logit_regression_problem import LogitRegressionProblem
from functions.quadratic_function import QuadraticFunction
from utils.utils import check_subset, load_common_problems


@ray.remote
class RayAgent:
    """
    Ray agent with remote communication and computation. But no algorithm is implemented. Algorithm should be a thing to
    reorganize the functions in ray agent. And the basic functions, such as grad_desc, aggregation, broadcast, and so on
    should be implemented in the RayAgent. In the other words, only basic actions are included in RayAgent, and how to
    organize the actions are organized in the algorithm. This also helpful when writing the synchronized algorithms.
    NOTICE: when create agents, should provide the name in options:
    RayAgent(...).options(name=f"{agent_id}").remote().
    after create the instances, agent_setup() should be firstly executed.
    """
    def __init__(
            self, agent_id: int, _network: Network, prob_settings: Dict[str, Any],
            alg_settings: Dict[str, Any], rst_settings: Dict[str, Any],
            param: np.ndarray = None, blk_indexes: Dict[int, List[int]] = None, wt_plan: str = "metropolis-Hastings"
    ) -> None:
        self.agent_id = agent_id
        np.random.seed(agent_id)  # setup random seed.
        self.agent_type = self._check_agent_type(_network.agent_types[agent_id])

        self._network = _network
        self.byz_size = _network.byz_size
        self.in_nbr_ids = _network.get_in_neighbors(agent_id)
        self.out_nbr_ids = _network.get_out_neighbors(agent_id)

        self.cached_in_nbr_handles = {}
        self.cached_out_nbr_handles = {}

        self.inbox = {in_nbr_id: {"param": None, "iter": 0} for in_nbr_id in self.in_nbr_ids}
        self.outbox = {out_nbr_id: {"param": None, "iter": 0} for out_nbr_id in self.out_nbr_ids}

        self._setup_problem(prob_settings)
        self.param: np.ndarray = param if isinstance(param, np.ndarray) else np.zeros(prob_settings['dimension'])
        self._setup_loss()
        self.M = self.prob.prob_settings['A']['HessUpperBd']
        self.m = self.prob.prob_settings['A']['HessLowerBd']

        self.alg_settings = alg_settings
        if self.alg_settings.get("byz_msg"):
            self.byz_plan = self.alg_settings['byz_msg']['byz_plan']
            self.byz_mag = self.alg_settings['byz_msg']['byz_mag']
        else:  # in original config files, there is no such setting.
            self.byz_plan = 'rand'  # default attack plan.
            self.byz_mag = 1  # default attack magnitude.

        self.rst_settings = rst_settings
        # TODO: change the following setup. the setup for the alg_settings['alg_name'] is not elegant enough.
        self.blk_indexes = blk_indexes if blk_indexes is not None else {0: list(range(self._network.net_size))}
        self.blk_num, self.blk_size = len(self.blk_indexes), len(self.blk_indexes[0])
        if alg_settings['alg_name'] == 'BBAlgorithm':
            self.loc_blk_wts = {blk_num_id: np.zeros(self._network.net_size) for blk_num_id in range(self.blk_num)}
        elif alg_settings['alg_name'] == 'DGDAlgorithm':
            self.loc_wts = self.generate_weights(weights_plan=wt_plan)[:, agent_id]  # different from block weights.

        self.end_iter = self.alg_settings['end_iter']
        self.hist_param = np.zeros((self.end_iter, self.prob.prob_settings['dimension']))

    def agent_setup(self) -> None:
        """This should be the first function called after __init__."""
        self._set_cached_handles()
        msgs = {out_nbr_id: {'param': self.param} for out_nbr_id in self.out_nbr_ids} if self.agent_type == 'Reg'\
            else {out_nbr_id: {'param': np.random.randn(*self.prob.dimension)} for out_nbr_id in self.out_nbr_ids}
        self._set_outbox(msgs)

    def _setup_problem(self, prob_settings: Dict[str, Any]) -> None:
        if prob_settings['load_problem']:  # load the problems from agent_id.
            load_problems_filename = prob_settings['problem_filename']
            self._load_problem(load_problems_filename)
        else:  # generate the problem for agent_id.
            self._gene_problem(prob_settings)

    def _gene_problem(self, prob_settings):
        if prob_settings['problem_type'] == 'LeastSquareProblem':
            self.prob = LeastSquareProblem(agent_id=self.agent_id, prob_settings=prob_settings)
            # self.loss = QuadraticFunction(self.prob.A, self.prob.b, self.prob.c, self.param)
        elif prob_settings['problem_type'] == 'LogitRegressionProblem':
            self.prob = LogitRegressionProblem(agent_id=self.agent_id, prob_settings=prob_settings)
            # self.loss = TODO: not finished for logit regression problem.
        else:
            raise ValueError(f"Invalid problem type: {prob_settings['problem_type']}")

    def _load_problem(self, common_problems_filename: str) -> None:
        common_problems = load_common_problems(common_problems_filename)
        self.prob = common_problems[self.agent_id]

    def _setup_loss(self) -> None:
        if self.prob.prob_settings['problem_type'] == 'LeastSquareProblem':
            self.loss = QuadraticFunction(self.prob.A, self.prob.b, self.prob.c, self.param)
        elif self.prob.prob_settings['problem_type'] == 'LogitRegressionProblem':
            raise NotImplementedError("Not finished for logit regression problem.")
        else:
            raise ValueError(f"Invalid problem type: {self.prob.prob_settings['problem_type']}")

    def local_filter_scalar(self, dim: int) -> List[int]:
        """
        Filter out the largest and smallest extreme values based on a specified dimension.
        self._network.byz_size: int
        return the agent_ids that are in the middle.
        """
        # Retrieve the parameters from the inbox and include the current agent's parameters.
        id_params: Dict[int, np.ndarray] = self.load_loc_params()
        # Sort the agents based on the specified dimension in their parameters.
        sorted_agents = sorted(id_params.items(), key=lambda item: item[1][dim])
        sorted_ids = [agent_id for agent_id, _ in sorted_agents]
        loc_index = sorted_ids.index(self.agent_id)
        # # Select the IDs to remove the highest and lowest values, taking into account the byz_size.
        sel_ids = sorted_ids[min(loc_index, self.byz_size):max(loc_index + 1, len(sorted_ids) - self.byz_size)]
        return sel_ids  # id of selected agents among dimension dim

    def local_filter_block(self, block: List[int]) -> List[int]:
        """
        Filter out the largest and smallest extreme values based on a specified dimension.
        args:
            block: dimensions in the block for the filter.
        return:
            the agent_ids that all the dimensions within the block are in the middle.
        """
        if not block:  # If block is empty, return an empty list
            raise ValueError("The blocksize is zero? Why this occur?")
            # return []
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

    def recv_from_nbr(self, sender_id: int, msg: Any) -> Tuple[int, Dict[int, Any]]:
        """Passive receive, used in send_to_nbrs. Save received information to inbox."""
        self.inbox[sender_id] = msg
        return sender_id, msg

    def broadcast(self) -> None:
        if self.agent_type == 'Reg':
            msgs = {out_nbr_id: {"param": self.param} for out_nbr_id in self.out_nbr_ids}
        elif self.agent_type == 'Byz':
            msgs = self._byz_msg_generator()
        else:
            msgs = {out_nbr_id: {"param": "something wrong."} for out_nbr_id in self.out_nbr_ids}
        self._set_outbox(msgs)
        self.send_to_nbrs(used_out_nbr_ids=self.out_nbr_ids)

    def full_aggregation(self) -> np.ndarray:
        tmp_param = np.sum([self.loc_wts[in_nbr_id] * self.inbox[in_nbr_id]['param']
                            for in_nbr_id in self.in_nbr_ids], axis=0)
        return self.loc_wts[self.agent_id] * self.param + tmp_param

    def block_byz_aggregation(self) -> np.ndarray:
        """
        block aggregation,
        return the full vector with blockwise aggregation through blockwise weights.
        """
        # step 0: generate the weights (related to blocks and iterations)
        self.loc_blk_wts = self.generate_block_weights(weights_plan='block_byz')  # default plan.
        tmp_param = np.zeros(self.prob.dimension)
        loc_params = self.load_loc_params()  # self included.
        for dim in range(self.prob.dimension):
            for blk_num_id in range(self.blk_num):
                for agent_j in self.blk_indexes[blk_num_id]:
                    tmp_param[dim] += self.loc_blk_wts[blk_num_id][agent_j]*loc_params[agent_j][dim]
        return tmp_param

    def atc_grad_desc(self, lr: float, cur_iter: int, agg_plan: str = "full_aggregation") -> None:
        """lr should be provided, should be part of algorithm."""
        # aggregation
        if agg_plan == 'full_aggregation':
            self.param = self.full_aggregation()
        elif agg_plan == 'block_byz_aggregation':
            self.param = self.block_byz_aggregation()
        else:
            raise ValueError(f"Unknown aggregation plan: {agg_plan}")
        # gradient update
        self.param = self.param - lr * self.loss.gd(self.param)
        # save to hist_param for save and plot
        self.hist_param[cur_iter] = self.param  # used for save and plot.

    def cta_grad_desc(self, lr: float, cur_iter: int, agg_plan: str = "full_aggregation") -> None:
        """lr should be provided, should be part of algorithm."""
        if agg_plan == 'full_aggregation':
            self.param = self.full_aggregation() - lr * self.loss.gd(self.param)
        elif agg_plan == "block_byz_aggregation":
            self.param = self.block_byz_aggregation() - lr * self.loss.gd(self.param)
        # save to hist_param for save and plot
        self.hist_param[cur_iter] = self.param

    def generate_weights(self, weights_plan: str = 'metropolis-Hastings') -> np.ndarray:
        # bb algorithm use another ways to generate weights for each block.
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
        """ Generate weights for each block for each iteration."""
        # print("\rStep 0.0: generate_block_weights: select plan for generate the block weights.\r")
        if weights_plan == 'block_byz':
            return self._generate_block_byz_weights()
        else:
            raise ValueError(f"Unknown __block__ weights plan: {weights_plan}.")

    def _generate_block_byz_weights(self) -> Dict[int, np.ndarray]:
        # print(f"\rStep 0.1: _generate_block_byz_weights: generate the block byzantine weights with block_byz plan.\r")
        block_wts = {}
        # print(f"The block_indexes information: {self.blk_indexes}")
        # print(f"\rNotice: current params are: {self.load_loc_params()} and with self: {self.param}")
        for blk_num_id, blk_index in self.blk_indexes.items():  # blk_id (in range(blk_num)) which block, blk_i
            # print(f"\rStep 0.1.0: _generate_block_byz_weights: The {blk_num_id}-th block "
            #       f"with index {blk_index} for the parameters.")
            sel_ids = self.local_filter_block(blk_index)
            # print(f"\rStep 0.1.1: _generate_block_byz_weights: "
            #       f"The selected agents for {blk_num_id}-th block are {sel_ids}.")
            # self_included = self._network.get_in_neighbors(self.agent_id)
            # self_included.append(self.agent_id)
            # check_subset(sel_ids, self_included)
            # print(f"\rStep 0.1.2: _generate_block_byz_weights: generate weight for the {blk_num_id}-th block.")
            block_wts[blk_num_id] = self._generate_one_block_byz_weight(sel_ids)  # just a row vector, not matrix.
        return block_wts

    def _generate_one_block_byz_weight(self, sel_ids: List[int]) -> np.ndarray:
        """sel_ids is related to iteration."""
        # print(f"\rStep 0.1.2.0: _generate_one_block_byz_weight: generate the weight.")
        net_size = self._network.net_size
        wx = 1/net_size  # lower bound of weight for x variable.
        # The weight can be changed to other values such as 1/(self._network.get_out_neighbors(self.agent_id)).
        reg_size = net_size - self.byz_size
        gamma = 0.5*(1+1/(1-self.m/(self.M+self.m)))  # self.M+self.m should be self.M, but use the sum for safe.
        wd = np.max([(reg_size-1-gamma)/(reg_size-2), wx])  # diagonal of weight should be positive.
        # print(f"\rStep 0.1.2.1: _generate_one_block_byz_weight: diagonal is {wd}.")
        # check_subset([self.agent_id], sel_ids)
        w_others = (1-wd) / (len(sel_ids)-1)  # sum of weights without agent_id itself.
        loc_wt_row = np.zeros(net_size)
        loc_wt_row[sel_ids] = w_others
        loc_wt_row[self.agent_id] = wd  # replace the `diagonal` one with its value.
        # print(f"\rStep 0.1.2.2: _generate_one_block_byz_weight: "
        #       f"check the {loc_wt_row}'s sum is {np.sum(np.array(loc_wt_row))} is row stochastic.")
        return loc_wt_row

    def _byz_msg_generator(self) -> Dict[int, Dict[str, np.ndarray]]:
        if self.agent_type != 'Byz':
            raise ValueError(f"Regular agent {self.agent_id} should not generate Byzantine messages.")
        if self.byz_plan == "rand":
            msgs = {out_nbr_id: {"param": self.byz_mag*np.random.rand(*self.param.shape)} for out_nbr_id in self.out_nbr_ids}
        elif self.byz_plan == "flip":
            msgs = {out_nbr_id: {"param": -self.byz_mag*self.param} for out_nbr_id in self.out_nbr_ids}
        elif self.byz_plan == "time":
            msgs = {out_nbr_id: {"param": self.byz_mag*self.param} for out_nbr_id in self.out_nbr_ids}
        elif self.byz_plan == "ones":
            msgs = {out_nbr_id: {"param": self.byz_mag*np.ones_like(self.param)} for out_nbr_id in self.out_nbr_ids}
        else:
            raise ValueError(f"Current byzantine plan: {self.byz_plan} is not supported.")
        return msgs

    @staticmethod
    def _check_agent_type(agent_type: str) -> str:
        if agent_type not in ['Reg', 'Byz']:
            raise ValueError(f"Invalid agent type: {agent_type}")
        return agent_type

    def _set_outbox(self, msgs: Dict[int, Any]) -> None:
        self.outbox = msgs

    def load_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

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

    def get_hist_param(self):
        return self.hist_param
