import ray
# from ray.exceptions import RayActorError

# from configs.configs import *
from networks.network import Network
from problems.least_square_problem import LeastSquareProblem
from utils.utils import parse_args, centralized_solver
from algorithms.bcgd_algorithm import BCGDAlgorithm
from algorithms.bb_algorithm import BBAlgorithm
from algorithms.dgd_algorithm import DGDAlgorithm
from algorithms.fcgd_algorithm import FCGDAlgorithm
algorithms = {
    "BBAlgorithm": BBAlgorithm,
    "BCGDAlgorithm": BCGDAlgorithm,
    "DGDAlgorithm": DGDAlgorithm,
    "FCGDAlgorithm": FCGDAlgorithm,
}

# load the parameters
args = parse_args()
print(args)
# print(args)
# generate the common network
common_network = Network(net_settings=args.net_settings)
# TODO: save the file with specialized name
# TODO: load the saved file
# generate the common problems
common_problems = LeastSquareProblem.generate_common_problems(
    net_size=args.net_settings['net_size'], prob_settings=args.prob_settings
)
print(f"The centralized solution is: {centralized_solver(common_problems, common_network.agent_types)}")
# TODO: save and load the common_problems
# TODO: load the saved common_problems

# This is a good way, but limited by the computation ability of the computer, too many agents, leads to unexpected dead.
# ray.init()
# # run all the algorithms in the alg_settings.
# for alg_name in args.alg_settings['alg_name']:
#
#     AlgorithmClass = algorithms[alg_name]
#     alg_args = args
#     alg_args.alg_settings['alg_name'] = alg_name
#     alg = AlgorithmClass(alg_args, common_network, common_problems)
#     alg.run()
#
#     # clean all agents for reset.
#     agent_handles = {agent_id: ray.get_actor(f"{agent_id}") for agent_id in range(args.net_settings['net_size'])}
#     for agent_id, agent_handle in agent_handles.items():
#         ray.kill(agent_handle)
#         # print(f"Agent {agent_id} has been killed.")
#
# ray.shutdown()

# run all the algorithms in the alg_settings.
for alg_name in args.alg_settings['alg_name']:
    ray.init()
    AlgorithmClass = algorithms[alg_name]
    alg_args = args
    alg_args.alg_settings['alg_name'] = alg_name
    alg = AlgorithmClass(alg_args, common_network, common_problems)
    alg.run()

    ray.shutdown()
