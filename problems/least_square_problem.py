# problems/least_square_problem.py

import numpy as np
import cvxpy as cvx
from typing import Any, Dict, Tuple
from problems.abstract_problem import AbstractProblem
import os
import joblib

from utils.utils import project_root, common_problems_dir


class LeastSquareProblem(AbstractProblem):
    """Least square problem with quadratic objective loss.
    objective loss: 0.5*x^T A x + b^T x + c
    """
    def __init__(self, agent_id: int, prob_settings: Dict[str, Any]) -> None:
        super().__init__(agent_id, prob_settings)
        seed = prob_settings.get("seed", 0)
        np.random.seed(seed)

        self.M, self.m = prob_settings['A']['HessUpperBd'], prob_settings['A']['HessLowerBd']
        self.b_cond = prob_settings['b']
        self.c = prob_settings['c']

        self.A, self.b = self.generate_abc()
        self.sol = self.closed_form_sol()

    def generate_abc(self) -> Tuple[np.ndarray, np.ndarray]:
        a_mat = self.generate_a(lbd=self.m, ubd=self.M, dim=self.dimension)
        b = self.generate_b(self.b_cond, self.dimension)
        return a_mat, b

    def closed_form_sol(self) -> np.ndarray:
        if self.constrain is None:
            return -np.linalg.inv(self.A) @ self.b
        else:
            raise NotImplementedError('Closed form solution is not implemented for constrained LS problems.')

    def cvx_sol(self) -> np.ndarray:
        if self.constrain is None:
            x = cvx.Variable(self.dimension)
            objective = cvx.Minimize(0.5 * cvx.quad_form(x, self.A) + self.b.T @ x + self.c)
            prob = cvx.Problem(objective)
            prob.solve()
            return x.value
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

    @classmethod
    def generate_common_problems(cls, net_size: int, prob_settings: Dict[str, Any]) -> Dict[int, 'LeastSquareProblem']:
        """common_problems = LeastSquareProblem.generate_common_problems(net_size, prob_settings) """
        if prob_settings['problem_type'] == 'LeastSquareProblem':
            common_problems = {
                agent_id: cls(agent_id=agent_id, prob_settings=prob_settings) for agent_id in range(net_size)
            }
        elif prob_settings['problem_type'] == 'LeastSquareProblem':
            raise NotImplementedError('LogitRegressionProblem is not finished yet.')
        else:
            raise ValueError(f"Invalid problem type: {prob_settings['problem_type']}")
        return common_problems

    @staticmethod
    def save_common_problems(common_problems: Dict[int, 'LeastSquareProblem'], filename: str) -> None:
        save_path = os.path.join(common_problems_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # generate folder if not exist
        joblib.dump(common_problems, save_path)
        print(f"Saved common problems to: {save_path}")

    @staticmethod
    def load_common_problems(filename: str) -> Dict[int, 'LeastSquareProblem']:
        load_path = os.path.join(common_problems_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No such file: '{load_path}'")
        common_problems = joblib.load(load_path)
        return common_problems
