# utils/utils.py

import argparse
import yaml
import time
from contextlib import contextmanager


def parse_args():
    parser = argparse.ArgumentParser(description="Run RayAgent optimization, setup for network, problem and algorithm")
    parser.add_argument(
        "-n", "--net", type=str, default="configs/test_ER_network_config.yml",
        help="Path to the network configuration file (default: configs/test_ER_network_config.yml)."
    )
    parser.add_argument(
        "-p", "--prob", type=str, default="configs/test_LSProblem_config.yml",
        help="Path to the problem configuration file (default: configs/test_LSProblem_config.yml)."
    )
    parser.add_argument(
        '-a', "--alg", type=str, default="configs/test_dgd_alg_config.yml",
        help="Path to the algorithm configuration file (default: configs/test_dgd_alg_config.yml)."
    )
    parser.add_argument(
        "-r", "--rst", type=str, default="configs/test_result_config.yml",
        help="Path to the result configuration file (default: configs/test_result_config.yml)."
    )

    args = parser.parse_args()

    network_config = _load_yaml(args.net)
    problem_config = _load_yaml(args.prob)
    algorithm_config = _load_yaml(args.alg)
    result_config = _load_yaml(args.rst)

    _merge_configs(args, [network_config, problem_config, algorithm_config, result_config])

    return args


@contextmanager
def timer():
    """
    with timer():
        code to test.
    """
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.6f} seconds")


def _load_yaml(file_path):
    """Load a YAML file and return its content as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def _merge_configs(args, configs):
    """Merge multiple configuration dictionaries into the argparse namespace."""
    for config in configs:
        for key, value in config.items():
            setattr(args, key, value)
