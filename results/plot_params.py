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

# byz_145 = [
#     "20240718_083936_result_base_end1000_ss1e3_net30_byz1_dim10_dgd.h5",
#     # "20240718_085952_result_base_end1000_ssdim_net30_byz1_dim10_dgd.h5",
#     "20240718_080005_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bcgd.h5",
#     "20240718_081947_result_base_end1000_ss1e3_net30_byz1_dim10_blksz2_bb.h5",
#     "20240718_122346_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_blksz2_bb.h5",
#     "20240718_123633_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_blksz2_bcgd.h5",
#     "20240718_124918_result_2024-07-18_end700_ss1e2_net17_byz4_dim10_dgd.h5",
#     "20240718_112503_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_blksz2_bb.h5",
#     "20240718_113754_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_blksz2_bcgd.h5",
#     "20240718_115042_result_2024-07-18_end700_ss1e2_net17_byz5_dim10_dgd.h5",
#     # "20240718_120327_result_2024-07-18_end700_ssdim_net17_byz5_dim10_dgd.h5",
# ]
# plot_distance_to_optimal(byz_145)

# byz1_init100 = [
#     "20240718_154011_result_2024-07-18_end700_ss1e2_net17_byz1_dim10_blksz3_init100_bb.h5",
#     "20240718_155312_result_2024-07-18_end700_ss1e2_net17_byz1_dim10_blksz3_init100_bcgd.h5",
#     "20240718_160618_result_2024-07-18_end700_ss1e2_net17_byz1_dim10_init100_dgd.h5",
#     "20240718_161922_result_2024-07-18_end700_ssdim_net17_byz1_dim10_init100_dgd.h5"
# ]
# plot_distance_to_optimal(byz1_init100)

# byz2_init100 = [
#     "20240718_074956_result_2024-07-18_end700_ss1e2_net17_byz2_dim10_blksz3_init100_bb.h5",
#     "20240718_080415_result_2024-07-18_end700_ss1e2_net17_byz2_dim10_blksz3_init100_bcgd.h5",
#     "20240718_081833_result_2024-07-18_end700_ss1e2_net17_byz2_dim10_init100_dgd.h5",
#     "20240718_083247_result_2024-07-18_end700_ssdim_net17_byz2_dim10_init100_dgd.h5"
# ]
# plot_distance_to_optimal(byz2_init100)

byz3_AHU2 = [
    "20240719_002829_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bb.h5",
    "20240719_004319_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bcgd.h5",
    "20240719_005750_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_init100_AHU2_dgd.h5",
    # "20240719_011226_result_2024-07-19_end700_ssdim_net17_byz3_dim10_init100_AHU2_dgd.h5"
]  # The DGD is better than bb and bcgd.

plot_distance_to_optimal(byz3_AHU2)