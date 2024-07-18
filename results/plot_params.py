from utils.utils import plot_distance_to_optimal

# byz0_filenames = [
#     "20240717_173944_result_base_end1000_ss1e3_net30_byz0_dim7_blksz2_bcgd.h5",
#     "20240717_175918_result_base_end1000_ss1e3_net30_byz0_dim7_blksz2_bb.h5",
#     "20240717_181905_result_base_end1000_ss1e3_net30_byz0_dim7_dgd.h5",
#     "20240717_183838_result_base_end1000_ssdim_net30_byz0_dim7_dgd.h5"
# ]
#
# plot_distance_to_optimal(byz0_filenames)
#
#
# byz1_filenames = [
#     "20240717_185808_result_base_end1000_ss1e3_net30_byz1_dim7_blksz2_bcgd.h5",
#     "20240717_191801_result_base_end1000_ss1e3_net30_byz1_dim7_blksz2_bb.h5",
#     "20240717_193816_result_base_end1000_ss1e3_net30_byz1_dim7_dgd.h5",
#     # "20240717_195823_result_base_end1000_ssdim_net30_byz1_dim7_dgd.h5"
# ]
#
# plot_distance_to_optimal(byz1_filenames)


# byz1_dim10_filenames = [
#     "20240718_080005_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bcgd.h5",
#     "20240718_081947_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bb.h5",
#     "20240718_083936_result_base_end1000_ss1e3_net30_byz1_dim10_dgd.h5",
#     "20240718_085952_result_base_end1000_ssdim_net30_byz1_dim10_dgd.h5"
# ]  # 1 byz agent
# plot_distance_to_optimal(byz1_dim10_filenames)

# blksz_2345_filenames = [
#     "20240718_080005_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bcgd.h5",
#     "20240718_081947_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bb.h5",
#     "20240718_092010_result_base_end1000_ss1e3_net30_byz1_dim10_blksz3_bcgd.h5",
#     "20240718_094003_result_base_end1000_ss1e3_net30_byz1_dim10_blksz3_bb.h5",
#     "20240718_095941_result_base_end1000_ss1e3_net30_byz1_dim10_blksz4_bcgd.h5",
#     "20240718_101920_result_base_end1000_ss1e3_net30_byz1_dim10_blksz4_bb.h5",
#     "20240718_103901_result_base_end1000_ss1e3_net30_byz1_dim10_blksz5_bcgd.h5",
#     "20240718_105841_result_base_end1000_ss1e3_net30_byz1_dim10_blksz5_bb.h5"
# ]  # larger block size leads to better performance.
# plot_distance_to_optimal(blksz_2345_filenames)

byz_145 = [
    "20240718_083936_result_base_end1000_ss1e3_net30_byz1_dim10_dgd.h5",
    "20240718_085952_result_base_end1000_ssdim_net30_byz1_dim10_dgd.h5",
    "20240718_080005_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bcgd.h5",
    "20240718_081947_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bb.h5",
    "20240718_122346_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_blksz2_bb.h5",
    "20240718_123633_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_blksz2_bcgd.h5",
    "20240718_124918_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_dgd.h5",
    "20240718_112503_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_blksz2_bb.h5",
    "20240718_113754_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_blksz2_bcgd.h5",
    "20240718_115042_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_dgd.h5",
    # "20240718_120327_result_2024-07-18_end700_ssdim_net17_byz5_dim10_dgd.h5",
]
plot_distance_to_optimal(byz_145)

