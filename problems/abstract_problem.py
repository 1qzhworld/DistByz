# problems/abstract_problem.py
from typing import Any, Literal


class AbstractProblem:
    def __init__(self, agent_id: int, problem_type: Literal['LeastSquare', 'LogitRegression'],
                 dimension: int, constrain: Any = None) -> None:
        self.agent_id = agent_id
        self.problem_type = problem_type
        self.dimension = dimension
        self.constrain = self.set_constrain(constrain) if constrain is not None else None

    def set_constrain(self, constrain: Any) -> Any:
        raise NotImplementedError("Constrain handling is not implemented.")