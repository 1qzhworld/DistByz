# communicators/ray_communicator.py
import logging

import numpy as np
import ray
# import dill
from typing import List, Any, Dict, Tuple


@ray.remote
class RayCommunicator:
    """Communicator class that performs communications through Ray."""

    def __init__(self, agent_id: int, in_nbr_ids: List[int], out_nbr_ids: List[int]):
        """
        Args:
            agent_id: name/id of the agent
            in_nbr_ids: in-neighbors
            out_nbr_ids: out-neighbors
        Params:
            self.inbox: {agent_id(int): message}
            self.outbox: {agent_id(int): message}
            self.agent_id: name, i.e., the id of agent
            self.in_nbr_ids: in-neighbors
            self.out_nbr_ids: out-neighbors
            self.cached_in_handles: {agent_id: ray.actor.ActorHandle} id to ActorHandle
            self.cached_out_handles: {agent_id: ray.actor.ActorHandle}
        """
        self.inbox = {in_nbr_id: {"param": None, "iter": 0} for in_nbr_id in in_nbr_ids}
        self.outbox = {out_nbr_id: {"param": None, "iter": 0} for out_nbr_id in out_nbr_ids}
        self.agent_id = agent_id
        self.in_nbr_ids = in_nbr_ids
        self.out_nbr_ids = out_nbr_ids
        self.cached_in_handles = {}
        self.cached_out_handles = {}

    def set_cached_neighbors(self):
        """
        Should be used immediately after RayCommunicators' init. Important for the future functions.
        """
        self.cached_in_handles = {in_nbr: ray.get_actor(f"{in_nbr}") for in_nbr in self.in_nbr_ids}
        self.cached_out_handles = {out_nbr: ray.get_actor(f"{out_nbr}") for out_nbr in self.out_nbr_ids}

    def get_ids(self):
        return self.agent_id, self.in_nbr_ids, self.out_nbr_ids

    def get_inbox(self):
        return self.inbox

    def get_outbox(self):
        return self.outbox

    def get_cached_in_handles(self):
        return self.cached_in_handles

    def get_cached_out_handles(self):
        return self.cached_out_handles

    def recv_from_nbr(self, sender_id: int, msg: Any) -> Tuple[int, Any]:
        """Passive receive a message from a neighbor. Used in send_xxx function(s).
        Args:
            sender_id: the id of the sender agent. This should not be ray.actor.ActorHandle.\
            Otherwise, the sender need to know its own ActorHandle which is not recommended in ray.
            msg: the message sent from sender agent.
        """
        self.inbox[sender_id] = msg
        return sender_id, msg

    def set_outbox(self, msgs: Dict[int, Any]):
        self.outbox = msgs  # can be used in self.get_outbox and send_to_nbrs

    def send_to_nbrs(self, used_out_nbr_ids: List[int], timeout: float = 1.0) -> None:
        """
        Send to out-neighbors with possibly different message. Timeouts and failures are recorded in log.
        Args:
            used_out_nbr_ids: can be a subset of self.out_nbr_ids
            timeout: timeout when the ray.wait is used.
        """
        self.check_subset(used_out_nbr_ids, self.out_nbr_ids)  # maybe only send to part of out-neighbors
        
        futures = [self.cached_out_handles[out_nbr_id].recv_from_nbr.remote(sender_id=self.agent_id, msg=msg)
                   for out_nbr_id, msg in self.outbox.items()]
        remaining_futures = futures
        while remaining_futures:
            ready_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=timeout)
            if not ready_futures:  # when ready_future is [] (empty).
                logging.info(f"Timeout reached on {self.agent_id}, some neighbors did not respond in time {timeout}s.")
                break
            for future in ready_futures:
                try:  # ensure the message is delivered.
                    ray.get(future)
                except Exception as e:
                    logging.info(f"Agent {self.agent_id} failed to send data to a neighbor: {e}")

    @staticmethod  # should I change this to a wrapper and demonstrate among the functions used this?
    def check_subset(short_list, long_list):
        if not set(short_list).issubset(set(long_list)):
            raise RuntimeError("Send or receive to/from invalid neighbors.")

    # def send_to_nbr(self, recv_id: int) -> Tuple[int, Any]:
    #     """
    #     return: sender_id and the message.
    #     """
    #     if recv_id not in self.out_nbr_ids:
    #         logging.info('Invalid access!')
    #     return self.agent_id, self.outbox[recv_id]

    # def recv_from_nbrs(self, used_in_nbr_ids: List[int], timeout: float = 1.0) -> Dict[int, Any]:
    #     """
    #     Positive receive from in-neighbors by getting from its outbox.
    #     Args:
    #         used_in_nbr_ids: can be a subset of self.in_nbr_ids (not check all)
    #         timeout: timeout when ray.wait is used.
    #     Return:
    #         {in_nbr_ids(int): message}
    #     """
    #     self.check_subset(used_in_nbr_ids, self.in_nbr_ids)
    #     received_data = {}  # {agent_id(int): message}
    #
    #     # used_cached_in_nbr_handles = {self.cached_in_handles[used_in_nbr] for used_in_nbr in used_in_nbr_ids}
    #     futures = {in_nbr_id: self.cached_in_handles[in_nbr_id].send_to_nbr.remote(
    #         [self.agent_id], self.outbox[self.agent_id], timeout)
    #         for in_nbr_id in used_in_nbr_ids}
    #     reverse_futures = {future: in_nbr_id for in_nbr_id, future in futures.items()}
    #
    #     while futures:
    #         ready_futures, remaining_futures = ray.wait(list(futures.values()), num_returns=1, timeout=timeout)
    #         if not ready_futures:
    #             print(f"Timeout reached, some neighbors did not respond in time {timeout}s.")
    #             break
    #         for ready_future in ready_futures:
    #             ready_in_nbr_id = reverse_futures[ready_future]
    #             try:
    #                 result = ray.get(ready_future)
    #                 if result:
    #                     in_nbr_id, msg = result
    #                     received_data[in_nbr_id] = msg
    #             except Exception as e:
    #                 logging.info(f"Failed to receive data from neighbor {ready_in_nbr_id}: {e}")
    #
    #             del futures[ready_in_nbr_id]
    #             del reverse_futures[ready_future]
    #     return received_data  # This should be the same as the self.inbox after send_to_nbrs is called.

    # def neighbors_exchange(self, send_obj: Any, in_neighbors: List[ray.actor.ActorHandle],
    #                        out_neighbors: List[ray.actor.ActorHandle], dict_neigh: bool = False) -> Dict[str, Any]:
    #     if not dict_neigh:
    #         self.send_to_nbrs(send_obj, out_neighbors)
    #     else:
    #         for j in out_neighbors:
    #             self.send_to_nbrs(send_obj[j], [j])
    #
    #     data = self.neighbors_receive(in_neighbors)
    #     return data
