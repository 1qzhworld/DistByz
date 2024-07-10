# tests/test_ray_agent.py

import ray
import numpy as np
import networkx as nx
from networks.network import Network
from problems.least_square_problem import LeastSquareProblem
from agents.ray_agent import RayAgent


def create_test_network():
    net_settings = {
        'conn_req': True,
        'directed_req': False,
        'ER_p': 0.5  # Probability for edge creation in ER graph
    }
    net_size = 3
    network = Network(net_size=net_size, gene_plan='ER_graph', net_settings=net_settings)
    return network


def test_ray_agent():
    ray.init(ignore_reinit_error=True)

    network = create_test_network()
    prob_settings = {
        'problem_type': 'LeastSquareProblem',
        'dimension': 2,  # Dimension of the problem
        'A': {
            'HessLowerBd': 1.0,
            'HessUpperBd': 10.0
        },
        'b': [-5, 5],  # Example vector b
        'c': 0  # Example scalar c
    }

    # agents = []
    # for agent_id in range(network.net_size):
    #     agent = RayAgent.options(name=f"{agent_id}").remote(agent_id=agent_id, agent_type='Reg', _network=network,
    #                                                         prob_settings=prob_settings)
    #     agents.append(agent)
    agents = [RayAgent.options(name=f"{agent_id}").remote(agent_id=agent_id, agent_type='Reg', _network=network,
                                                          prob_settings=prob_settings)
              for agent_id in range(network.net_size)]

    # Set up agents
    setup_futures = [agent.agent_setup.remote() for agent in agents]
    ray.get(setup_futures)

    # Perform gradient descent on each agent
    grad_desc_futures = [agent.grad_desc.remote(lr=0.1) for agent in agents]
    ray.get(grad_desc_futures)

    # Broadcast parameters to neighbors
    broadcast_futures = [agent.broadcast.remote() for agent in agents]
    ray.get(broadcast_futures, timeout=15.0)
    # ready_bc, remaining_bc = ray.wait(broadcast_futures, num_returns=len(broadcast_futures), timeout=10.0)
    # if remaining_bc:
    #     print("???")

    # Aggregate parameters from neighbors
    aggregation_futures = [agent.aggregation.remote() for agent in agents]
    ray.get(aggregation_futures)

    # Fetch parameters for verification
    params = ray.get([agent.get_param.remote() for agent in agents])
    print("Parameters after one iteration of gradient descent, broadcast, and aggregation:")
    for i, param in enumerate(params):
        print(f"Agent {i}: {param}")

    ray.shutdown()


if __name__ == "__main__":
    test_ray_agent()