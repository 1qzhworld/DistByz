# configs/configs.py

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
os.environ['PROJECT_ROOT'] = project_root
sys.path.append(project_root)

algorithms_dir = os.path.join(project_root, 'algorithms')
algorithms_filename = [
    filename[:-3] for filename in os.listdir(algorithms_dir)
    if filename.endswith("_algorithm.py")
]

configs_dir = os.path.join(project_root, 'configs')
data_dir = os.path.join(project_root, 'data')
common_problems_dir = os.path.join(data_dir, 'common_problems')
common_networks_dir = os.path.join(data_dir, 'common_networks')
functions_dir = os.path.join(data_dir, 'functions')
logs_dir = os.path.join(project_root, 'logs')
results_dir = os.path.join(project_root, 'results')
tests_dir = os.path.join(project_root, 'tests')
utils_dir = os.path.join(project_root, 'utils')

# GenePlan = Literal['ER_graph', 'aver_deg', 'reg_graph', 'geo_graph', 'load_fixed']
# AgentType = Literal['Reg', 'Byz']


