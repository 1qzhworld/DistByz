import ray
from configs.configs import *
from networks.network import Network
from problems.least_square_problem import LeastSquareProblem
from utils.utils import parse_args
from algorithms.bcgd_algorithm import BCGDAlgorithm
from algorithms.bb_algorithm import BBAlgorithm
from algorithms.dgd_algorithm import DGDAlgorithm
algorithms = {
    "BBAlgorithm": BBAlgorithm,
    "BCGDAlgorithm": BCGDAlgorithm,
    "DGDAlgorithm": DGDAlgorithm
}

# load the parameters
args = parse_args()
# print(args)
# generate the common network
common_network = Network(net_settings=args.net_settings)
# TODO: save the file with specialized name
# TODO: load the saved file
# generate the common problems
common_problems = LeastSquareProblem.generate_common_problems(
    net_size=args.net_settings['net_size'], prob_settings=args.prob_settings
)
# TODO: save and load the common_problems
# TODO: load the saved common_problems
ray.init()
#
# run all the algorithms in the alg_settings.
for alg_name in args.alg_settings['alg_name']:
    AlgorithmClass = algorithms[alg_name]
    alg_args = args
    alg_args.alg_settings['alg_name'] = alg_name
    alg = AlgorithmClass(alg_args, common_network, common_problems)
    alg.run()


ray.shutdown()
