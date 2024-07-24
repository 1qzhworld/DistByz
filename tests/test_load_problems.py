import os
import unittest
from utils.utils import generate_common_problems, save_common_problems, load_common_problems


class TestProblemLoading(unittest.TestCase):

    def setUp(self):
        self.prob_settings = {
            'load_problem': True,
            'seed': 0,
            'problem_type': 'LeastSquareProblem',
            'dimension': 10,
            'A': {
                'HessLowerBd': 1.0,
                'HessUpperBd': 10.0,  # AHU in config title,
            },
            'b': [-50, 50],
            'c': 0.0,
        }

        self.filename = 'test_common_problem_utils.joblib'
        self.common_problems = generate_common_problems(net_size=3, prob_settings=self.prob_settings)
        save_common_problems(common_problems=self.common_problems, filename=self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_load_problems(self):
        loaded_problems = load_common_problems(filename=self.filename)

        # Check that the loaded problems match the saved ones
        self.assertEqual(len(loaded_problems), len(self.common_problems))
        for original, loaded in zip(self.common_problems, loaded_problems):
            self.assertEqual(original, loaded, "Loaded problem does not match the saved problem")

    def test_problem_attributes(self):
        loaded_problems = load_common_problems(filename=self.filename)

        # Check that the loaded problems match the saved ones
        for original, loaded in zip(self.common_problems, loaded_problems):
            # Assuming LeastSquareProblem has an __eq__ method implemented
            self.assertEqual(original, loaded, "Loaded problem does not match the saved problem")


if __name__ == '__main__':
    unittest.main()