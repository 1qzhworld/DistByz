# results/plot_selected.py

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatter
from configs.configs import logs_dir, results_dir
from typing import List, Dict
import os
import numpy as np
from utils.utils import load_all_hist, get_reg_params


plt.rcParams['text.usetex'] = True


def final_plot(end_iter: int = None, save_path: str = None):
    plt.figure()
    ###################################################################################################################
    distance = cal_dist(
        "209_03H00M14S_result_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1_DGDAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='DGD Atk 20', linewidth=5, color='0.7', alpha=0.7, linestyle='-.')

    distance = cal_dist(
        "209_14H11M17S_result_209_10H28M31S_blk3_wtAV_init10_ones40_byz5_N50L10D9ss1e1_DGDAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='DGD Atk 40', linewidth=1, color='0', alpha=0.7, linestyle='-.')
    ###################################################################################################################
    distance = cal_dist(
        "209_03H00M14S_result_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1_FCGDAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='FCGD Atk 20', linewidth=5, color='0.7', alpha=0.7, linestyle='--')

    distance = cal_dist(
        "209_14H11M17S_result_209_10H28M31S_blk3_wtAV_init10_ones40_byz5_N50L10D9ss1e1_FCGDAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='FCGD Atk 50', linewidth=1, color='0', alpha=0.7, linestyle='--')
    ###################################################################################################################
    distance = cal_dist(
        "209_03H00M14S_result_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1_CWTMAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='CWTM Atk 20', linewidth=5, color='0.7', alpha=0.7, linestyle=':')

    distance = cal_dist(
        "209_14H11M17S_result_209_10H28M31S_blk3_wtAV_init10_ones40_byz5_N50L10D9ss1e1_CWTMAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='CWTM Atk 50', linewidth=1, color='0', alpha=0.7, linestyle=':')
    ###################################################################################################################

    distance = cal_dist(
        # "209_03H09M36S_result_209_10H28M31S_blk2_wtAV_init10_ones100_byz5_N50L10D9ss5e2_BBAlgorithm.h5", end_iter
        "209_03H00M14S_result_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1_BBAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='BCGD Atk 20', linewidth=5, color='0.7', alpha=0.7, linestyle='-')

    distance = cal_dist(
        "209_14H11M17S_result_209_10H28M31S_blk3_wtAV_init10_ones40_byz5_N50L10D9ss1e1_BBAlgorithm.h5", end_iter
    )
    plt.semilogy(range(end_iter), distance, label='BCGD Atk 50', linewidth=1, color='0', alpha=0.7, linestyle='-')

    plt.xlim(0, end_iter)
    ax = plt.gca()
    ax.set_ylabel('Sum of Distance from All Regular States to $x^*$', labelpad=0)

    log_formatter = LogFormatter(base=10, labelOnlyBase=False)
    ax.yaxis.set_major_formatter(log_formatter)

    # 手动设置要显示的 y 轴刻度
    y_ticks = [2e2, 3e2, 4e2, 6e2, 1e3]
    ax.set_yticks(y_ticks)

    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment('center')

    # 强制重新绘制刻度标签以应用自定义格式
    ax.figure.canvas.draw()

    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)

    plt.show()


def cal_dist(filename: str, end_iter: int = None):
    logging_result_path = os.path.join(logs_dir, filename)
    all_params, global_solution, agent_types = load_all_hist(logging_result_path)
    reg_params = get_reg_params(all_params, agent_types)
    num_iterations = end_iter if end_iter else all_params[0].shape[0]
    # distance = cal_dist(reg_params, global_solution, end_iter)
    distance = []
    for i in range(end_iter):
        squared_diff = 0
        for params in reg_params.values():
            squared_diff += np.linalg.norm(params[i] - global_solution)
            # squared_diff += np.sum((params[i] - global_solution) ** 2)
        distance.append(squared_diff)
    return distance


if __name__ == '__main__':
    save_name = "209_15H10M18S_selected_plots.pdf"
    path = os.path.join(results_dir, save_name)
    final_plot(30, save_path=path)
