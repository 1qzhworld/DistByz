# utils/utils.py

import argparse
import os
import ray
import yaml
import time
from contextlib import contextmanager
from typing import List, Any

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


def parse_args():
    parser = argparse.ArgumentParser(description="Run RayAgent optimization, setup for network, problem and algorithm")
    parser.add_argument(
        "-n", "--net", type=str, default=f"{project_root}/configs/test_ER_network_config.yml",
        help="Path to the network configuration file (default: configs/test_ER_network_config.yml)."
    )
    parser.add_argument(
        "-p", "--prob", type=str, default=f"{project_root}/configs/test_LSProblem_config.yml",
        help="Path to the problem configuration file (default: configs/test_LSProblem_config.yml)."
    )
    parser.add_argument(
        '-a', "--alg", type=str, default=f"{project_root}/configs/test_dgd_alg_config.yml",
        help="Path to the algorithm configuration file (default: configs/test_dgd_alg_config.yml)."
    )
    parser.add_argument(
        "-r", "--rst", type=str, default=f"{project_root}/configs/test_result_config.yml",
        help="Path to the result configuration file (default: configs/test_result_config.yml)."
    )

    parser.add_argument(
        "-o", "--one", type=str, default=f"{project_root}/configs/test_all_in_one.yml",
        help="Path to the unified configuration file (default: None)."
    )

    args = parser.parse_args()

    # Initialize configurations
    network_config = {}
    problem_config = {}
    algorithm_config = {}
    result_config = {}
    # Load unified configuration if provided (higher priority than split configs.)
    if args.one:
        # print(f"here we are in {args.one}")
        unified_config = _load_yaml(args.one)
        network_config = unified_config.get('network', {})
        # print(f"Our network config: {network_config}")
        problem_config = unified_config.get('problem', {})
        algorithm_config = unified_config.get('algorithm', {})
        result_config = unified_config.get('result', {})

    # Override with specific configuration files if all in one is not provided
    if not args.net:
        network_config.update(_load_yaml(args.net))
    if not args.prob:
        problem_config.update(_load_yaml(args.prob))
    if not args.alg:
        algorithm_config.update(_load_yaml(args.alg))
    if not args.rst:
        result_config.update(_load_yaml(args.rst))

    _merge_configs(args, [network_config, problem_config, algorithm_config, result_config])

    return args


def _load_yaml(file_path):
    """Load a YAML file and return its content as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def _merge_configs(args, configs):
    """Merge multiple configuration dictionaries into the argparse namespace."""
    for config in configs:
        for key, value in config.items():
            setattr(args, key, value)


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


@contextmanager
def ray_context():
    """with ray_context():"""
    ray.init()
    try:
        yield
    finally:
        ray.shutdown()


def check_subset(sublist: List[Any], super_list: List[Any]):
    if not set(sublist).issubset(set(super_list)):
        raise RuntimeError("Send or receive to/from invalid neighbors.")


if __name__ == "__main__":
    args = parse_args()
    print(args.net_settings)
