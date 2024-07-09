# tests/test_ray_agent.py

import numpy as np
import networkx as nx
import pytest
from problems.least_square_problem import LeastSquareProblem
from networks.network import Network
from functions.quadratic_function import QuadraticFunction
from agents.ray_agent import RayAgent

@pytest.fixture
def network():
    net_settings = {
        'conn_req': True,
        'directed_req': False,
        'ER_p': 0.5
    }
    return Network(net_size=4, gene_plan='ER_graph', net_settings=net_settings)

@pytest.fixture
def problem():
    agent_id = 0
    dimension = 5
    prob_info = {
        'A': {'HessLowerBd': 1.0, 'HessUpperBd': 10.0},
        'b': (-5, 5),
        'c': 0.0
    }
    return LeastSquareProblem(agent_id=agent_id, problem_type='LeastSquareProblem',
                              dimension=dimension, prob_info=prob_info)

@pytest.fixture
def quadratic_function():
    dimension = 5
    a = np.random.rand(dimension, dimension)
    a = a.T @ a  # 确保 a 是正定的
    b = np.random.rand(dimension, 1)
    c = np.random.rand(1)
    x_init = np.random.rand(dimension, 1)
    return QuadraticFunction(a, b, c, x_init)

@pytest.fixture
def ray_agent(network, problem, quadratic_function):
    agent_id = 0
    return RayAgent.remote(agent_id=agent_id, agent_type='Reg', network=network, problem=problem, function=quadratic_function)

def test_ray_agent_optimize(ray_agent):
    ray_agent.optimize()
    solution = ray_agent.get_solution()
    assert solution is not None, "The optimization did not return a solution."
    assert isinstance(solution, np.ndarray), "The solution is not a numpy array."

def test_ray_agent_solution(ray_agent):
    solution = ray_agent.get_solution()
    assert solution is not None, "The initial solution is None."
    assert isinstance(solution, np.ndarray), "The initial solution is not a numpy array."