# algorithms/dgd_algorithm.py

import ray
import numpy as np
from typing import Dict, List, Literal

from algorithms.abstract_algorithm import AbstractAlgorithm
from networks.network import Network
from problems.abstract_problem import AbstractProblem
from communicators.ray_communicator import RayCommunicator

AgentType = Literal['Reg', 'Byz']


class DGDAlgorithm(AbstractAlgorithm):
    def __init__(self, agent_id: int, agent_type: AgentType, network: Network,
                 prob: AbstractProblem, comm: RayCommunicator, alg_settings: dict, result_setting: dict):
        super().__init__(agent_id, agent_type, network, prob, comm, alg_settings, result_setting)

    def run(self, verbose: bool = False):
        while self.iter < self.end_iter:
            # setup stepsize for self.iter
            self.lr = self.stepsize(self.iter) if callable(self.stepsize) else self.stepsize
            # receive information from in-neighbors
            self.in_nbr_msgs = ray.get(self.comm.get_inbox.remote())
            # run one update
            self.iter_run()
            # broadcast the updated parameters
            self.broadcast()

            # print the update information
            if verbose:
                print(f'\r Agent {self.agent_id} is running {self.iter} iter of DGD algorithm.\r')
                # print(f'Agent {self.agent_id} has parameter {self.agent_param} at {self.iter} iter.\r')
                # print(f'Agent {self.agent_id} has buffer {self.buff} at {self.iter} iter.\r')

            self.iter += 1
            if self.res_func_eval:
                self.func_eval[self.iter - 1] = self.loss.eval(self.agent_param)
            if self.res_dist_to_opt:
                self.dist2opt[self.iter - 1] = np.linalg.norm(self.agent_param - self.lc_opt)

    def iter_run(self):
        self.aggregation()
        self.grad_desc()

    def grad_desc(self):
        self.agent_param = self.agent_param - self.lr * self.loss.gd(self.agent_param)

    def aggregation(self):
        tmp_param = np.sum([self.loc_wt[in_nbr] * self.in_nbr_msgs[in_nbr]["param"]
                            for in_nbr in self.in_nbr_ids], axis=0)
        self.agent_param = self.loc_wt[self.agent_id] * self.agent_param + tmp_param

    def broadcast(self):
        if self.agent_type == 'Reg':
            msgs = {out_nbr_id: {"param": self.agent_param, "iter": self.iter}
                    for out_nbr_id in self.out_nbr_ids}
        else:  # for byzantine agents
            # TODO: generate different attack types.
            msgs = {out_nbr_id: {"param": np.random.rand(self.agent_param.shape), "iter": self.iter}
                    for out_nbr_id in self.out_nbr_ids}
        self.comm.set_outbox(msgs)
        self.comm.send_to_nbrs.remote(self.out_nbr_ids)
