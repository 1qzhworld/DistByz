# utils/utils.py

import argparse
import os
import re
import warnings
import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from typing import List, Any, Dict

from datetime import datetime
from configs.configs import project_root, logs_dir, configs_dir

np.random.seed(0)


def parse_args(config_filename=None):
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
        "-a", "--all", type=str,
        default=config_filename or "test_207_07H14M29S_new.yml",
        help="Path to the unified configuration file (default: None)."
    )
    # parser.add_argument(
    #     "-o", "--one", type=str, default=f"{project_root}/configs/test_base_end2000_ssdim_net30_byz0_dim7_dgd.yml",
    #     help="Path to the unified configuration file (default: None)."
    # )

    args = parser.parse_args()
    current_timestamp = datetime.now().strftime("%j_%HH%MM%SS")
    args.timestamp = current_timestamp  # add timestamp here such that the files have same prefixes.

    # Initialize configurations
    network_config = {}
    problem_config = {}
    algorithm_config = {}
    result_config = {}
    # print(args.all)
    # Load unified configuration if provided (higher priority than split configs.)
    if args.all:
        print(f"Training with config file: {args.all}")
        unified_config = _load_yaml(os.path.join(configs_dir, args.all))
        network_config = unified_config.get('network', {})
        problem_config = unified_config.get('problem', {})
        algorithm_config = unified_config.get('algorithm', {})
        result_config = unified_config.get('result', {})

    _merge_configs(args, [network_config, problem_config, algorithm_config, result_config])
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


def generate_log_path(
        config_path: str, timestamp: str, ext: str, logging_dir: str = logs_dir, filename_plan: str = 'config', **kwargs
) -> str:
    if ext not in ['yml', 'yaml', 'h5']:
        raise ValueError(f"ext should be one of 'yml', 'yaml', 'h5'. Current ext: {ext}, is not implemented.")
    if filename_plan == 'config':
        if ext not in ['yml', 'yaml']:
            raise ValueError(f"ext should be one of 'yml', 'yaml' for config. Current ext: {ext}, is not implemented.")
        config_name = os.path.basename(config_path)
        logging_config_name = os.path.splitext(config_name)[0].replace('test', 'result')
        new_filename = f"{timestamp}_{logging_config_name}.{ext}"
        return os.path.join(logging_dir, new_filename)
    elif filename_plan == 'alg_hist':
        if ext != 'h5':
            raise ValueError(f"ext should be one of 'h5', not {ext}.")
        if 'alg_name' not in kwargs:
            raise ValueError(f"alg_name should be one of kwargs: {kwargs.keys()}. Not provided yet.")
        alg_name = kwargs['alg_name']
        config_name = os.path.basename(config_path)
        logging_alg_hist_name = os.path.splitext(config_name)[0].replace('test', 'result')
        new_filename = f"{timestamp}_{logging_alg_hist_name}_{alg_name}.{ext}"
        return os.path.join(logging_dir, new_filename)
    else:
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
    longest_common_substring_in_result_files = find_longest_common_substring_in_all(filenames)
    plt.title(longest_common_substring_in_result_files)
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


def longest_common_substring(str1, str2):
    m, n = len(str1), len(str2)
    result = 0
    end = 0
    length = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                length[i][j] = length[i - 1][j - 1] + 1
                if length[i][j] > result:
                    result = length[i][j]
                    end = i - 1
            else:
                length[i][j] = 0

    return str1[end - result + 1:end + 1]


def find_longest_common_substring_in_all(filenames):
    if not filenames:
        return ""

    # Start with the first string
    common_substr = filenames[0]

    # Compare with each subsequent string and update the common substring
    for i in range(1, len(filenames)):
        common_substr = longest_common_substring(common_substr, filenames[i])
        # If at any step the common substring is empty, return immediately
        if not common_substr:
            return ""

    return common_substr


def extract_parameter(filename_parts, prefix):
    for part in filename_parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


# def check_configurations(config_path):
#     config_filename = os.path.basename(config_path)
#     filename_parts = config_filename.split('_')
#
#     end_iter_str = extract_parameter(filename_parts, 'end')
#     stepsize_str = extract_parameter(filename_parts, 'ss')
#     net_size_str = extract_parameter(filename_parts, 'net')
#     byz_size_str = extract_parameter(filename_parts, 'byz')
#     dimension_str = extract_parameter(filename_parts, 'dim')
#     blk_size_str = extract_parameter(filename_parts, 'blksz')
#     param_init_str = extract_parameter(filename_parts, 'init')
#
#     byz_plan_and_mag_str = extract_parameter(filename_parts, 'plan')
#     if byz_plan_and_mag_str:
#         # Extract plan and magnitude
#         byz_plan_str = ''.join(filter(str.isalpha, byz_plan_and_mag_str))
#         byz_mag_str = ''.join(filter(str.isdigit, byz_plan_and_mag_str))
#     else:
#         byz_plan_str = None
#         byz_mag_str = None
#
#     if end_iter_str is None:
#         warnings.warn(f"Parameter 'end' not found in the filename: {config_filename}")
#     else:
#         end_iter = int(end_iter_str)
#
#     if stepsize_str is None:
#         warnings.warn(f"Parameter 'ss' not found in the filename: {config_filename}")
#     else:
#         if stepsize_str == 'dim':
#             stepsize = 'diminishing_stepsize'
#         else:
#             stepsize = float(stepsize_str.replace('e', 'e-'))
#
#     if net_size_str is None:
#         warnings.warn(f"Parameter 'net' not found in the filename: {config_filename}")
#     else:
#         net_size = int(net_size_str)
#
#     if byz_size_str is None:
#         warnings.warn(f"Parameter 'byz' not found in the filename: {config_filename}")
#     else:
#         byz_size = int(byz_size_str)
#
#     if dimension_str is None:
#         warnings.warn(f"Parameter 'dim' not found in the filename: {config_filename}")
#     else:
#         dimension = int(dimension_str)
#
#     if blk_size_str is None:
#         if "bb" in config_filename or "bcgd" in config_filename:
#             warnings.warn(f"Parameter 'blksz' required but not found in the filename: {config_filename}")
#     else:
#         blk_size = int(blk_size_str)
#
#     if param_init_str is None:
#         warnings.warn(f"Parameter 'param_init' not found in the filename: {config_filename}")
#     else:
#         param_init = int(param_init_str)
#
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#
#     alg_settings = config['algorithm']['alg_settings']
#     net_settings = config['network']['net_settings']
#     prob_settings = config['problem']['prob_settings']
#     byz_msg = config['algorithm']['alg_settings']['byz_msg']
#
#     errors = []
#
#     if end_iter_str is not None and alg_settings['end_iter'] != end_iter:
#         errors.append(f"end_iter: expected {end_iter}, found {alg_settings['end_iter']}")
#
#     if stepsize_str is not None:
#         if stepsize == 'diminishing_stepsize':
#             if alg_settings['stepsize'] != 'diminishing_stepsize':
#                 errors.append(f"stepsize: expected diminishing_stepsize, found {alg_settings['stepsize']}")
#         else:
#             if alg_settings['stepsize'] != stepsize:
#                 errors.append(f"stepsize: expected {stepsize}, found {alg_settings['stepsize']}")
#
#     if net_size_str is not None and net_settings['net_size'] != net_size:
#         errors.append(f"net_size: expected {net_size}, found {net_settings['net_size']}")
#
#     if byz_size_str is not None and net_settings['byz_size'] != byz_size:
#         errors.append(f"byz_size: expected {byz_size}, found {net_settings['byz_size']}")
#
#     if dimension_str is not None and prob_settings['dimension'] != dimension:
#         errors.append(f"dimension: expected {dimension}, found {prob_settings['dimension']}")
#
#     if blk_size_str is not None and alg_settings.get('blk_size') is not None and alg_settings['blk_size'] != blk_size:
#         errors.append(f"blk_size: expected {blk_size}, found {alg_settings['blk_size']}")
#
#     if param_init_str is not None and alg_settings.get('param_init') is not None and alg_settings[
#         'param_init'] != param_init:
#         errors.append(f"param_init: expected {param_init}, found {alg_settings['param_init']}")
#
#     if byz_plan_str is not None and byz_msg.get('byz_plan') != byz_plan_str:
#         errors.append(f"byz_plan: expected {byz_plan_str}, found {byz_msg.get('byz_plan')}")
#
#     if byz_mag_str is not None:
#         byz_mag = int(byz_mag_str)
#         if byz_msg.get('byz_mag') != byz_mag:
#             errors.append(f"byz_mag: expected {byz_mag}, found {byz_msg.get('byz_mag')}")
#
#     if errors:
#         raise ValueError(f"Configurations in {config_filename} do not match its name:\n" + "\n".join(errors))


def find_a_key_by_value(d, target_value):  # can be used to find the agent_id of Byz.
    for key, value in d.items():
        if value == target_value:
            return key
    return None


def centralized_solver(common_problems, agent_types) -> np.ndarray:
    reg_problems = [common_problems[agent_id] for agent_id in range(len(agent_types)) if agent_types[agent_id] == 'Reg']
    sum_a = np.sum([prob.A for prob in reg_problems], axis=0)
    sum_b = np.sum([prob.b for prob in reg_problems], axis=0)
    return (-np.linalg.inv(0.5 * (sum_a.T + sum_a)) @ sum_b).flatten()

def main():
    # parser = argparse.ArgumentParser(description="Utility script")
    # parser.add_argument('--check-config', type=str, help='Check the configuration file against its filename')
    # args = parser.parse_args()
    #
    # if args.check_config:
    #     try:
    #         check_configurations(args.check_config)
    #         print(f"{args.check_config} matches its configuration.")
    #     except (ValueError, FileNotFoundError) as e:
    #         print(str(e))
    #         exit(1)
    pass


if __name__ == "__main__":
    main()
