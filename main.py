import ray

from utils.utils import parse_args, generate_common_problems
from algorithms import bb_algorithm, bcgd_algorithm, dgd_algorithm
algorithms = {
    "BBAlgorithm": bb_algorithm,
    "BCGDAlgorithm": bcgd_algorithm,
    "DGDAlgorithm": dgd_algorithm
}

args = parse_args()
# generate the common network
common_problems = generate_common_problems(net_size=args.net_size, prob_settings=args.prob_settings)

#


ray.init()

# run all the algorithms in the alg_settings.
for alg_name in args.alg_settings['alg_name']:
    alg = algorithms[alg_name](args)
    alg.run()


ray.shutdown()