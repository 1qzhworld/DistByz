import pytest
import numpy as np
from problems.least_square_problem import LeastSquareProblem


@pytest.fixture
def problem_params():
    return {
        "agent_id": 1,
        "problem_type": "LeastSquare",
        "dimension": 3,
        "prob_info": {
            "A": {
                "HessLowerBd": 1.0,
                "HessUpperBd": 10.0
            },
            "b": [-5, 5],
            "c": 0.0
        }
    }


def test_generate_abc(problem_params):
    problem = LeastSquareProblem(**problem_params)
    A, b, c = problem.generate_abc(problem_params['prob_info'])
    assert A.shape == (problem_params['dimension'], problem_params['dimension'])
    assert b.shape == (problem_params['dimension'], 1)
    assert isinstance(c, float)


def test_closed_form_solution(problem_params):
    problem = LeastSquareProblem(**problem_params)
    solution = problem.closed_form_sol()
    assert solution.shape == (problem_params['dimension'], 1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
