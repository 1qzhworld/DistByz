# algorithms/alg_dgd.py

import ray
import logging
# from typing import Dict, Any
# from problems.least_square_problem import LeastSquareProblem
from networks.network import Network
from agents.ray_agent import RayAgent
from utils.utils import parse_args, gene_byz


def main():
    args = parse_args()

    network = Network(net_size=args.net_size, gene_plan=args.gene_plan, net_settings=args.net_settings)
    # TODO set the weight for algorithm.
    agent_types = gene_byz(args.net_size, args.reg_size)

    alg_settings = args.alg_settings
    stepsize = alg_settings['stepsize']
    end_iter = alg_settings['end_iter']
    cur_iter = 0

    agents = {}
    logging.info("Start ray.")
    ray.init(ignore_reinit_error=True)
    logging.info('Start to setup agents.')
    for agent_id in range(network.net_size):
        agents[agent_id] = RayAgent.options(name=f'{agent_id}').remote(
            agent_id=agent_id, agent_type=agent_types[agent_id], _network=network, prob_settings=args.prob_info
        )
    comm_setup_futures = [agent.agent_setup.remote() for _, agent in agents.items()]
    setup_timeout = 20
    _, remaining_comm = ray.wait(comm_setup_futures, num_returns=len(comm_setup_futures), timeout=setup_timeout)
    if remaining_comm:
        raise TimeoutError(f"Timeout reached, {len(remaining_comm)} setup did not respond in time {setup_timeout}s.")
    logging.info('Successfully setup all agents.')
    print('Successfully setup all agents.')

    while cur_iter < end_iter:
        print(f'Current iteration: {cur_iter}.')
        bc_futures = [agent.broadcast.remote() for _, agent in agents.items()]
        _, remaining_bc = ray.wait(bc_futures, num_returns=len(bc_futures), timeout=setup_timeout)
        if remaining_bc:
            logging.info(f"Not all agents received broadcast at iteration {cur_iter}.")

        agg_futures = [agent.aggregation.remote() for _, agent in agents.items()]
        _, remaining_agg = ray.wait(agg_futures, num_returns=len(agg_futures), timeout=setup_timeout)
        if remaining_agg:
            logging.info(f"Not all agents aggregate successfully at iteration {cur_iter}.")

        lr = stepsize(iter) if callable(stepsize) else stepsize
        upd_futures = [agent.grad_desc.remote(lr) for _, agent in agents.items()]
        _, remaining_upd = ray.wait(upd_futures, num_returns=len(upd_futures), timeout=setup_timeout)
        if remaining_upd:
            logging.info(f"Not all agents update successfully at iteration {cur_iter}.")

        cur_iter += 1

    print(f"Agent 0's inbox is {agents[0].get_inbox.remote()}.")
    logging.info('Shutdown ray.')
    ray.shutdown()


if __name__ == "__main__":
    main()
