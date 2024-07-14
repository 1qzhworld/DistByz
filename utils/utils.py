# utils/utils.py

import argparse
import os

import numpy as np
import ray
import yaml
import time
from contextlib import contextmanager
from typing import List, Any, Dict

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


def parse_args(set_one=None):
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
        "-o", "--one", type=str, default=set_one or f"{project_root}/configs/test_all_in_one_bb.yml",
        help="Path to the unified configuration file (default: None)."
    )
    # parser.add_argument(
    #     "-o", "--one", type=str, default=f"{project_root}/configs/test_all_in_one_dgd.yml",
    #     help="Path to the unified configuration file (default: None)."
    # )

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


# def extract_dimensions(
#         inbox_params: Dict[int, np.ndarray], sel_ids: List[int], blk_index: List[int]
# ) -> Dict[int, np.ndarray]:
#     """
#     Removed: this is not necessary to extract dimensions for calculating the block weighted average.
#     Args:
#         inbox_params: parameters from inbox
#     Extract the selected agents (from sel_ids) parameters' specialized dimension (blk_index)
#     from the global parameters."""
#     sel_params = {}
#     check_subset(sel_ids, list(inbox_params.keys()))
#     for sel_id in sel_ids:
#         sel_params[sel_id] = inbox_params[sel_id]
#     sel_block_params = {}
#     for sel_id in sel_ids:
#         sel_block_params[sel_id] = sel_params[sel_id][blk_index[sel_id]]
#     # return sel_block_params
#     return {sel_id: inbox_params[sel_id][blk_idx] for sel_id, blk_idx in blk_index.items() if sel_id in sel_ids}

if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"\rThe key: {k} has value {v}.\r")
    print(args.net_settings)
