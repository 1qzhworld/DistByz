# problems/abstract_problem.py
from typing import Any, Dict


class AbstractProblem:
    def __init__(self, agent_id: int, prob_settings: Dict[str, Any]) -> None:
        self.agent_id = agent_id
        self.problem_type = prob_settings['problem_type']
        self.dimension = prob_settings['dimension']
        self.constrain = self.set_constrain(prob_settings['constrain']) if prob_settings.get('constrain') else None

    def set_constrain(self, constrain: Any) -> Any:
        raise NotImplementedError("Constrain handling is not implemented.")
