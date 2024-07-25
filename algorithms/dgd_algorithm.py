# algorithms/abstract_algorithm.py

# import sys
# import os

# # 获取脚本的当前目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 项目根目录（假设 utils 文件夹与 algorithms 文件夹在同一级）
# project_root = os.path.join(current_dir, '..')
# # 添加项目根目录到 sys.path
# sys.path.append(project_root)

from utils.utils import parse_args
from algorithms.abstract_algorithm import AbstractAlgorithm
# os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用Ray的日志去重


class DGDAlgorithm(AbstractAlgorithm):
    def __init__(self, args, common_network, common_problems):
        self.alg_name = self.__class__.__name__
        if args.alg_settings['alg_name'] != self.alg_name:
            raise ValueError(f"Setting for {args.alg_settings['alg_name']}, but here is {self.alg_name}.")
        super(DGDAlgorithm, self).__init__(args, common_network, common_problems)

    def iter_run(self):
        setup_timeout = self.timeout_base * self.network.net_size
        bc_futures = [agent_handle.broadcast.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(bc_futures, setup_timeout, "broadcast")

        agg_futures = [agent_handle.full_aggregation.remote() for agent_handle in self.agents.values()]
        self.wait_for_all_futures(agg_futures, setup_timeout, "aggregation")

        self.lr = self.gene_stepsize()

        # gradient descent update
        upd_futures = [agent_handle.atc_grad_desc.remote(self.lr, self.cur_iter, agg_plan='full_aggregation')
                       for agent_handle in self.agents.values()]
        # upd_futures = [agent_handle.cta_grad_desc.remote(self.lr, self.cur_iter, agg_plan='full_aggregation')
        #                for agent_handle in self.agents.values()]

        self.wait_for_all_futures(upd_futures, setup_timeout, "update")


# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(script_dir, '..'))
#     # set_one_config_path = f"{project_root}/configs/test_base_end2000_ss1e3_net30_byz0_dim7_dgd.yml"
#     # dgd_args = parse_args(set_one=set_one_config_path)
#     dgd_args = parse_args()
#     dgd_algorithm = DGDAlgorithm(dgd_args, )
#     dgd_algorithm.run()
