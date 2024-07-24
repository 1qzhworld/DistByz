# problems/abstract_problem.py
from typing import Any, Dict
import numpy as np


class AbstractProblem:
    def __init__(self, agent_id: int, prob_settings: Dict[str, Any]) -> None:
        self.agent_id = agent_id
        self.prob_settings = prob_settings
        self.problem_type = prob_settings['problem_type']
        self.dimension = prob_settings['dimension']
        self.constrain = self.set_constrain(prob_settings['constrain']) if prob_settings.get('constrain') else None
        # Notice: multi threading using same seed, the behavior may become unpredictable.
        self.seed = prob_settings.get('seed', 0) + agent_id
        np.random.seed(self.seed)

    def set_constrain(self, constrain: Any) -> Any:
        raise NotImplementedError("Constrain handling is not implemented.")
