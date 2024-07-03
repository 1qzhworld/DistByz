# problems/least_square_problem.py

import numpy as np
import cvxpy as cvx
from typing import Any, Dict, Tuple, Literal
from problems.abstract_problem import AbstractProblem


class LeastSquareProblem(AbstractProblem):
    """Least square problem with quadratic objective loss.
    objective loss: 0.5*x^T A x + b^T x + c
    """
    def __init__(self, agent_id: int, problem_type: Literal['LeastSquare'], dimension: int,
                 abc_info: Dict[str, Any], constrain: Any = None) -> None:
        super().__init__(agent_id, problem_type, dimension, constrain)
        self.A, self.b, self.c = self.generate_abc(abc_info)
        self.sol = self.closed_form_sol()

    def generate_abc(self, abc_conditions: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        A = self.generate_A(abc_conditions['A']['HessLowerBd'],
                            abc_conditions['A']['HessUpperBd'],
                            self.dimension)
        b = self.generate_b(abc_conditions['b'], self.dimension)
        c = abc_conditions['c']
        return A, b, c

    def closed_form_sol(self) -> np.ndarray:
        if self.constrain is None:
            return -np.linalg.inv(self.A) @ self.b
        else:
            raise NotImplementedError('Closed form solution is not implemented for constrained LS problems.')

    @staticmethod
    def generate_A(lbd: float, ubd: float, dim: int) -> np.ndarray:
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
