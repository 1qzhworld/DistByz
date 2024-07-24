# problems/abstract_problem.py

import os
import joblib

from typing import Any, Dict

from configs.configs import common_problems_dir


class AbstractProblem:
    def __init__(self, agent_id: int, prob_settings: Dict[str, Any]) -> None:
        self.agent_id = agent_id
        self.problem_type = prob_settings['problem_type']
        self.dimension = prob_settings['dimension']
        self.constrain = self.set_constrain(prob_settings['constrain']) if prob_settings.get('constrain') else None

    def set_constrain(self, constrain: Any) -> Any:
        raise NotImplementedError("Constrain handling is not implemented.")

    @classmethod
    def generate_common_problems(cls, net_size: int, prob_settings: Dict[str, Any]) -> Dict[int, 'AbstractProblem']:
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
    def save_common_problems(common_problems: Dict[int, 'AbstractProblem'], filename: str) -> None:
        save_path = os.path.join(common_problems_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # generate folder if not exist
        joblib.dump(common_problems, save_path)
        print(f"Saved common problems to: {save_path}")

    @staticmethod
    def load_common_problems(filename: str) -> Dict[int, 'AbstractProblem']:
        load_path = os.path.join(common_problems_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No such file: '{load_path}'")
        common_problems = joblib.load(load_path)
        return common_problems
