# main.py

import ray
import logging

from problems.least_square_problem import LeastSquareProblem
from networks.network import Network
from agents.ray_agent_old import RayAgent
from utils.utils import parse_args  # timer


def setup_network(args):
    return Network(net_size=args.net_size, gene_plan=args.gene_plan, net_settings=args.net_settings)


def setup_problem(args):
    return LeastSquareProblem(agent_id=1, problem_type=args.problem_type,
                              dimension=args.dimension, prob_info=args.prob_info)


def main():
    args = parse_args()  # default setting is stored in utils.parse_args()
    # print(args)
    network = setup_network(args)
    # problem = setup_problem(args)
    ray.init(ignore_reinit_error=True)
    ray_agents = {}
    for agent_id in range(network.net_size):
        ray_agents[agent_id] = RayAgent.remote(
            agent_id=agent_id, agent_type='Reg', network=network, problem_type=args.problem_type,
            dimension=args.dimension, prob_info=args.prob_info, alg_name="DGDAlgorithm", alg_settings=args.alg_settings,
            result_settings=args.result_settings
        )
    futures = [ray_agent.comm_setup.remote() for _, ray_agent in ray_agents.items()]
    ready_futures, remaining_futures = ray.wait(futures, num_returns=len(futures), timeout=2.0)
    if remaining_futures:
        logging.info(f"Timeout reached, some neighbors did not respond in time 2.0s.")

    comms = [ray.get(agent.get_comm.remote()) for _, agent in ray_agents.items()]
    algs = [ray.get(agent.get_alg.remote()) for _, agent in ray_agents.items()]
    probs = [ray.get(agent.get_prob.remote()) for _, agent in ray_agents.items()]

    [ray_agent.run.remote() for _, ray_agent in ray_agents.items()]
    ray.shutdown()
    logging.info(f"Agent 0 is running {algs[0].iter} iter of DGD algorithm.")

if __name__ == "__main__":
    main()
