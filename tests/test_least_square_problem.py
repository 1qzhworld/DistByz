# tests/test_least_square_problem.py

import pytest
import numpy as np
from problems.least_square_problem import LeastSquareProblem


def test_generate_a():
    dim = 3
    lbd = 1.0
    ubd = 5.0
    A = LeastSquareProblem.generate_a(lbd, ubd, dim)
    assert A.shape == (3, 3), "Matrix A should have shape (3, 3)"
    assert np.all(np.linalg.eigvals(A) >= lbd), "All eigenvalues should be >= lbd"
    assert np.all(np.linalg.eigvals(A) <= ubd), "All eigenvalues should be <= ubd"


def test_generate_b():
    dim = 3
    condition_b = (-1.0, 1.0)
    b = LeastSquareProblem.generate_b(condition_b, dim)
    assert b.shape == (dim, 1), "Vector b should have shape (dim, 1)"
    assert np.all(b >= condition_b[0]), "All elements of b should be >= condition_b[0]"
    assert np.all(b <= condition_b[1]), "All elements of b should be <= condition_b[1]"


def test_closed_form_sol():
    prob_settings = {
        'A': {
            'HessLowerBd': 1.0,
            'HessUpperBd': 5.0
        },
        'b': (-1.0, 1.0),
        'c': 0.0,
        'dimension': 3,
        'problem_type': 'LeastSquareProblem'
    }
    agent_id = 0
    lsp = LeastSquareProblem(agent_id, prob_settings)
    print(f'\r---------------{lsp}-------------------\r')
    sol = lsp.closed_form_sol()
    assert sol.shape == (prob_settings['dimension'], 1), "Solution vector should have shape (dimension, 1)"


def test_cvx_sol():
    prob_settings = {
        'A': {
            'HessLowerBd': 1.0,
            'HessUpperBd': 5.0
        },
        'b': (-1.0, 1.0),
        'c': 0.0,
        'dimension': 3
    }
    agent_id = 0
    lsp = LeastSquareProblem(agent_id, prob_settings)
    cvx_solution = lsp.cvx_sol()
    closed_form_solution = lsp.closed_form_sol()
    np.testing.assert_almost_equal(cvx_solution, closed_form_solution, decimal=5,
                                   err_msg="CVX solution should match closed form solution")


if __name__ == "__main__":
    pytest.main([__file__])
