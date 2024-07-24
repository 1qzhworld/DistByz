# configs/configs.py

import os
from typing import Literal

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

algorithms_dir = os.path.join(project_root, 'algorithms')
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
