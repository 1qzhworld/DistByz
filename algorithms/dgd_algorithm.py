# algorithms/abstract_algorithm.py

import sys
import os

# 获取脚本的当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（假设 utils 文件夹与 algorithms 文件夹在同一级）
project_root = os.path.join(current_dir, '..')
# 添加项目根目录到 sys.path
sys.path.append(project_root)

from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm
# os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用Ray的日志去重


class DGDAlgorithm(AbstractAlgorithm):
    def __init__(self, args):
        self.alg_name = "DGDAlgorithm"
        if args.alg_settings['alg_name'] != self.alg_name:
            raise ValueError(f"Setting for {args.alg_settings['alg_name']}, but here is DGDAlgorithm.")
        super(DGDAlgorithm, self).__init__(args)

    def iter_run(self):
        setup_timeout = self.timeout_base * self.network.net_size
        bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

        agg_futures = [agent_handle.aggregation.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(agg_futures, setup_timeout, "aggregation")

        self.lr = self.gene_stepsize()

        # gradient descent update
        upd_futures = [agent_handle.grad_desc.remote(self.lr) for agent_handle in self.agents.values()]
        self.wait_for_all_futures(upd_futures, setup_timeout, "update")


if __name__ == "__main__":
    dgd_args = parse_args()
    dgd_algorithm = DGDAlgorithm(dgd_args)
    dgd_algorithm.run()
