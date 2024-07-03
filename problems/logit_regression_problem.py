import numpy as np
from typing import Any, Literal
from problems.abstract_problem import AbstractProblem


class LogitRegression(AbstractProblem):
    def __init__(self, agent_id: int, problem_type: Literal['LogitRegression'],
                 dimension: int, constrain: Any = None) -> None:
        super().__init__(agent_id, problem_type, dimension, constrain)
        self.params = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        raise NotImplementedError("Fit method is not yet implemented.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Predict method is not yet implemented.")

    def set_constrain(self, constrain: Any) -> Any:
        raise NotImplementedError("Constrain handling is not implemented.")