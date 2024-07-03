import pytest
import numpy as np
import cvxpy as cp
from functions.quadratic_function import QuadraticFunction


def test_closed_form_minimizer():
    # Define the quadratic function parameters
    a = np.array([[3, 2], [2, 6]])
    b = np.array([-2, -4])
    c = np.array([0])
    x_init = np.array([0, 0])

    # Initialize the QuadraticFunction object
    quad_func = QuadraticFunction(a, b, c, x_init)

    # Get the closed form minimizer
    closed_form_solution = quad_func.closed_form_minimizer()

    # Define the optimization problem using cvxpy
    x = cp.Variable(len(b))
    objective = cp.Minimize(0.5 * cp.quad_form(x, a) + b.T @ x + c)
    prob = cp.Problem(objective)
    prob.solve()

    # Get the cvxpy solution
    cvxpy_solution = x.value
    print (x)

    # Assert that the solutions are close to each other
    np.testing.assert_allclose(closed_form_solution, cvxpy_solution, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
