# problems/least_square_problem.py

import numpy as np
import cvxpy as cvx
from typing import Any, Dict, Tuple
from problems.abstract_problem import AbstractProblem


class LeastSquareProblem(AbstractProblem):
    """Least square problem with quadratic objective loss.
    objective loss: 0.5*x^T A x + b^T x + c
    """
    def __init__(self, agent_id: int, prob_settings: Dict[str, Any]) -> None:
        super().__init__(agent_id, prob_settings)
        seed = prob_settings.get("seed", 0)
        np.random.seed(seed)
        self.A, self.b, self.c = self.generate_abc(prob_settings)
        self.sol = self.closed_form_sol()

    def generate_abc(self, abc_conditions: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        a_mat = self.generate_a(abc_conditions['A']['HessLowerBd'], abc_conditions['A']['HessUpperBd'], self.dimension)
        b = self.generate_b(abc_conditions['b'], self.dimension)
        c = abc_conditions['c']
        return a_mat, b, c

    def closed_form_sol(self) -> np.ndarray:
        if self.constrain is None:
            return -np.linalg.inv(self.A) @ self.b
        else:
            raise NotImplementedError('Closed form solution is not implemented for constrained LS problems.')

    @staticmethod
    def generate_a(lbd: float, ubd: float, dim: int) -> np.ndarray:
        assert ubd >= lbd >= 0, "Ensure that u >= l >= 0."
        eigenvalues = np.random.uniform(lbd, ubd, dim)
        diag = np.diag(eigenvalues)
        q, _ = np.linalg.qr(np.random.randn(dim, dim))
        return q @ diag @ q.T

    @staticmethod
    def generate_b(condition_b: Any, dim: int) -> np.ndarray:
        return np.random.uniform(condition_b[0], condition_b[1], (dim, 1))

    def cvx_sol(self) -> np.ndarray:
        if self.constrain is None:
            x = cvx.Variable(self.dimension)
            objective = cvx.Minimize(0.5 * cvx.quad_form(x, self.A) + self.b.T @ x + self.c)
            prob = cvx.Problem(objective)
            prob.solve()
            return x.value
        else:
            raise NotImplementedError('Closed form solution is not implemented for constrained LS problems.')
