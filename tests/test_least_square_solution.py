import pytest
import numpy as np
import cvxpy as cvx
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


def test_closed_form_vs_cvxpy_solution(problem_params):
    problem = LeastSquareProblem(**problem_params)
    closed_form_solution = problem.closed_form_sol().flatten()  # Flatten to match cvxpy_solution shape

    # Define the optimization problem using cvxpy
    x = cvx.Variable(problem_params["dimension"])
    objective = cvx.Minimize(0.5 * cvx.quad_form(x, problem.A) + problem.b.T @ x + problem.c)
    prob = cvx.Problem(objective)
    prob.solve()

    # Get the cvxpy solution
    cvxpy_solution = x.value

    # Assert that the solutions are close to each other
    np.testing.assert_allclose(closed_form_solution, cvxpy_solution, rtol=1e-15, atol=1e-15)


if __name__ == "__main__":
    pytest.main(["-v", __file__])