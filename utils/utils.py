# utils/utils.py

import argparse
import os
import ray
import yaml
import time
import random
import numpy as np
from contextlib import contextmanager
from typing import Dict, Literal, List, Any
from networks.network import Network

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

    args = parser.parse_args()

    network_config = _load_yaml(args.net)
    problem_config = _load_yaml(args.prob)
    algorithm_config = _load_yaml(args.alg)
    result_config = _load_yaml(args.rst)

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
