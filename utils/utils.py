# utils/utils.py

import argparse
import os
import re

import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Any, Dict


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

np.random.seed(0)


def parse_args(set_one=None):
    parser = argparse.ArgumentParser(description="Run RayAgent optimization, setup for network, problem and algorithm")
    # parser.add_argument(
    #     "-n", "--net", type=str, default=f"{project_root}/configs/test_ER_network_config.yml",
    #     help="Path to the network configuration file (default: configs/test_ER_network_config.yml)."
    # )
    # parser.add_argument(
    #     "-p", "--prob", type=str, default=f"{project_root}/configs/test_LSProblem_config.yml",
    #     help="Path to the problem configuration file (default: configs/test_LSProblem_config.yml)."
    # )
    # parser.add_argument(
    #     '-a', "--alg", type=str, default=f"{project_root}/configs/test_dgd_alg_config.yml",
    #     help="Path to the algorithm configuration file (default: configs/test_dgd_alg_config.yml)."
    # )
    # parser.add_argument(
    #     "-r", "--rst", type=str, default=f"{project_root}/configs/test_result_config.yml",
    #     help="Path to the result configuration file (default: configs/test_result_config.yml)."
    # )

    parser.add_argument(
        "-o", "--one", type=str,
        default=set_one or f"{project_root}/configs/test_base_end2000_ss1e3_net30_byz0_dim7_blksz2_bb.yml",
        help="Path to the unified configuration file (default: None)."
    )
    # parser.add_argument(
    #     "-o", "--one", type=str, default=f"{project_root}/configs/test_base_end2000_ssdim_net30_byz0_dim7_dgd.yml",
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
        print(f"here we are in {args.one}")
        unified_config = _load_yaml(args.one)
        network_config = unified_config.get('network', {})
        # print(f"Our network config: {network_config}")
        problem_config = unified_config.get('problem', {})
        algorithm_config = unified_config.get('algorithm', {})
        result_config = unified_config.get('result', {})

    # # Override with specific configuration files if all in one is not provided
    # if not args.net:
    #     network_config.update(_load_yaml(args.net))
    # if not args.prob:
    #     problem_config.update(_load_yaml(args.prob))
    # if not args.alg:
    #     algorithm_config.update(_load_yaml(args.alg))
    # if not args.rst:
    #     result_config.update(_load_yaml(args.rst))

    _merge_configs(args, [network_config, problem_config, algorithm_config, result_config])

    # Generate a filename for the result configuration with a timestamp
    # logging_dir = project_root+f"/logging/"
    logging_path = generate_logging_path(config_one_path=args.one, ext='yml', filename_plan='config_one')

    save_config(args, logging_path)

    return args


def _load_yaml(file_path):
    """Load a YAML file and return its content as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def _merge_configs(args: argparse.Namespace, configs: List[Dict[str, Any]]):
    """Merge multiple configuration dictionaries into the argparse namespace."""
    for config in configs:
        for key, value in config.items():
            setattr(args, key, value)


def generate_logging_path(config_one_path: str, ext: str, filename_plan: str = 'config_one') -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ext not in ['yml', 'yaml', 'h5']:
        raise ValueError(f"ext should be one of 'yml', 'yaml', 'h5'. Current ext: {ext}, is not implemented.")
    if filename_plan == 'config_one':
        logging_dir_name = os.path.join(project_root, "logging")
        config_one_name = os.path.basename(config_one_path)
        logging_config_one_name = os.path.splitext(config_one_name)[0].replace('test', 'result')
        new_filename = f"{timestamp}_{logging_config_one_name}.{ext}"
        return os.path.join(logging_dir_name, new_filename)
    else:
        # return os.path.splitext(filename_plan)[0].replace()
        raise NotImplementedError


def save_config(args: argparse.Namespace, saving_config_path: str) -> None:
    """
    Save the configuration dictionary to a YAML file in logging.
    """
    with open(saving_config_path, 'w') as file:
        yaml.dump(args, file)


def load_all_hist(logging_result_path):
    """Copied from AbstractAlgorithm.load_all_hist()
    Not used yet, for future restart training for neural networks and so on."""
    with h5py.File(logging_result_path, 'r') as f:
        load_all_params = {  # both regular and byzantine
            int(agent_id): f[agent_id][()] for agent_id in f.keys()
            if agent_id not in {"global_solution", "prob_a_list", "prob_b_list", "agent_ids", "agent_types_values"}
        }
        global_solution = f["global_solution"][()]
        prob_a_list = f["prob_a_list"][()]  # only regular
        prob_b_list = f["prob_b_list"][()]  # only regular
        agent_ids = f['agent_ids'][:]
        agent_types_values = f['agent_types_values'][:]
    agent_types = {int(agent_id): agent_type.decode('utf-8') for agent_id, agent_type in
                   zip(agent_ids, agent_types_values)}
    return load_all_params, global_solution, prob_a_list, prob_b_list, agent_types


def get_reg_params(all_params: Dict[int, np.ndarray], agent_types: Dict[int, str]):
    return {agent_id: all_params[agent_id] for agent_id in range(len(agent_types)) if agent_types[agent_id] == 'Reg'}


def plot_distance_to_optimal(filenames: List[str]):
    """filenames: example: ["20240716_112329_result_all_in_one_bcgd.h5", "20240716_113756_result_all_in_one_bb.h5"]"""
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(script_dir, '..'))
    logging_result_base = os.path.join(project_root, "logging")

    plt.figure()  # 创建一个新的绘图对象

    for filename in filenames:
        logging_result_path = os.path.join(logging_result_base, filename)
        all_params, global_solution, prob_a_list, prob_b_list, agent_types = load_all_hist(logging_result_path)
        reg_params = get_reg_params(all_params, agent_types)
        num_iterations = all_params[0].shape[0]

        distances = []
        for i in range(num_iterations):
            squared_diff = 0
            for params in reg_params.values():
                squared_diff += np.sum((params[i] - global_solution) ** 2)
            distances.append(squared_diff)

        # Extract algorithm name from filename
        algorithm_name = extract_algorithm_name(filename)

        # Plotting each line on the same figure
        plt.semilogy(range(num_iterations), distances, label=f'{algorithm_name} Distance to Optimal')

    plt.xlabel('Iteration')
    plt.ylabel('Squared Distance (Log Scale)')
    plt.title('Distance to Optimal Solution over Iterations for Multiple Algorithms')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


def extract_algorithm_name(filename: str) -> str:
    match = re.search(r'_([^_]+)\.h5$', filename)
    if match:  # 提取文件名中最后一个下划线后的部分，并转换为大写字母
        return match.group(1).upper()
    return "UNKNOWN"


def check_subset(sublist: List[Any], super_list: List[Any]):
    if not set(sublist).issubset(set(super_list)):
        raise RuntimeError("Send or receive to/from invalid neighbors.")


def check_configurations(config_path):
    config_filename = os.path.basename(config_path)

    filename_parts = config_filename.split('_')
    end_iter_str = filename_parts[2]
    stepsize_str = filename_parts[3]
    net_size_str = filename_parts[4]
    byz_size_str = filename_parts[5]
    dimension_str = filename_parts[6]

    end_iter = int(end_iter_str.replace('end', ''))
    net_size = int(net_size_str.replace('net', ''))
    byz_size = int(byz_size_str.replace('byz', ''))
    dimension = int(dimension_str.replace('dim', ''))

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    alg_settings = config['algorithm']['alg_settings']
    net_settings = config['network']['net_settings']
    prob_info = config['problem']['prob_info']

    errors = []

    if alg_settings['end_iter'] != end_iter:
        errors.append(f"end_iter: expected {end_iter}, found {alg_settings['end_iter']}")

    if stepsize_str == 'ssdim':
        if alg_settings['stepsize'] != 'diminishing_stepsize':
            errors.append(f"stepsize: expected diminishing_stepsize, found {alg_settings['stepsize']}")
    else:
        stepsize = float(stepsize_str.replace('ss', '').replace('e', 'e-'))
        if alg_settings['stepsize'] != stepsize:
            errors.append(f"stepsize: expected {stepsize}, found {alg_settings['stepsize']}")

    if net_settings['net_size'] != net_size:
        errors.append(f"net_size: expected {net_size}, found {net_settings['net_size']}")
    if net_settings['byz_size'] != byz_size:
        errors.append(f"byz_size: expected {byz_size}, found {net_settings['byz_size']}")
    if prob_info['dimension'] != dimension:
        errors.append(f"dimension: expected {dimension}, found {prob_info['dimension']}")

    requires_blk_size = "bb" in config_filename or "bcgd" in config_filename
    if requires_blk_size:
        blksz_str = filename_parts[7]
        blk_size = int(blksz_str.replace('blksz', ''))
        if alg_settings['blk_size'] != blk_size:
            errors.append(f"blk_size: expected {blk_size}, found {alg_settings['blk_size']}")

    if errors:
        raise ValueError(f"Configurations in {config_filename} do not match its name:\n" + "\n".join(errors))

def main():
    parser = argparse.ArgumentParser(description="Utility script")
    parser.add_argument('--check-config', type=str, help='Check the configuration file against its filename')
    args = parser.parse_args()

    if args.check_config:
        try:
            check_configurations(args.check_config)
            print(f"{args.check_config} matches its configuration.")
        except (ValueError, FileNotFoundError) as e:
            print(str(e))
            exit(1)


if __name__ == "__main__":
    main()
