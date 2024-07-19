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

# byz3_AHU2 = [
#     "20240719_002829_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_004319_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_005750_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_init100_AHU2_dgd.h5",
#     "20240719_011226_result_2024-07-19_end700_ssdim_net17_byz3_dim10_init100_AHU2_dgd.h5"
# ]  # The DGD is better than bb and bcgd.
#
# plot_distance_to_optimal(byz3_AHU2)


# byz2_AHU2 = [
#     # "20240719_002829_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bb.h5",
#     # "20240719_004319_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_blksz3_init100_AHU2_bcgd.h5",
#     # "20240719_005750_result_2024-07-19_end700_ss1e2_net17_byz3_dim10_init100_AHU2_dgd.h5",
#     # "20240719_011226_result_2024-07-19_end700_ssdim_net17_byz3_dim10_init100_AHU2_dgd.h5",
#     "20240719_024050_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_025532_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_032423_result_2024-07-19_end700_ssdim_net17_byz2_dim10_init100_AHU2_dgd.h5",
#     "20240719_030951_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_init100_AHU2_dgd.h5"
# ]  # 2 or three byzantine agents' influcence are similar. the lines are almost the same.
# plot_distance_to_optimal(byz2_AHU2)

# the number of byzantine agent seems do not influence the performance of the algorithm for agent 1,2,3 in the network
# process similar performances.
# byz1_AHU2 = [
# # "20240719_024050_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_blksz3_init100_AHU2_bb.h5",
# #     "20240719_025532_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_blksz3_init100_AHU2_bcgd.h5",
# #     "20240719_032423_result_2024-07-19_end700_ssdim_net17_byz2_dim10_init100_AHU2_dgd.h5",
# #     "20240719_030951_result_2024-07-19_end700_ss1e2_net17_byz2_dim10_init100_AHU2_dgd.h5",
#     "20240719_021148_result_2024-07-19_end700_ss1e2_net17_byz1_dim10_init100_AHU2_dgd.h5",
#     "20240719_014224_result_2024-07-19_end700_ss1e2_net17_byz1_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_015700_result_2024-07-19_end700_ss1e2_net17_byz1_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_022616_result_2024-07-19_end700_ssdim_net17_byz1_dim10_init100_AHU2_dgd.h5"
# ]
# plot_distance_to_optimal(byz1_AHU2)


# byz2_AHU2_flip9 = [
#     "20240719_124643_result_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_125926_result_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_131212_result_2024-07-19_end700_ss1e2_net17_byz2_planflip9_dim10_init100_AHU2_dgd.h5",
#     "20240719_132505_result_2024-07-19_end700_ssdim_net17_byz2_planflip9_dim10_init100_AHU2_dgd.h5"
# ]  # BCGD is better, BB is worst, DGD is fast in initialization, strange thing???
# plot_distance_to_optimal(byz2_AHU2_flip9)

# TODO: can be useful?
# byz2_AHU2_time9 = [
#     # "20240719_115547_result_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_blksz3_init100_AHU2_bb.h5",
#     # "20240719_120830_result_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_122114_result_2024-07-19_end700_ss1e2_net17_byz2_plantime9_dim10_init100_AHU2_dgd.h5",
#     "20240719_123356_result_2024-07-19_end700_ssdim_net17_byz2_plantime9_dim10_init100_AHU2_dgd.h5"
# ]   # dgd becomes overflow, out of range error.
# plot_distance_to_optimal(byz2_AHU2_time9)

# TODO: DGD withi diminishing stepsize converge too fase. And others are still decreasing. However, BCGD is best.
# byz2_AHU2_ones9 = [
#     "20240719_043804_result_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_045321_result_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_050800_result_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_init100_AHU2_dgd.h5",
#     # "20240719_052225_result_2024-07-19_end700_ssdim_net17_byz2_planones9_dim10_init100_AHU2_dgd.h5"
# ]
# plot_distance_to_optimal(byz2_AHU2_ones9)

# byz2_AHU2_rand9 = [
#     "20240719_034657_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_040150_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_041627_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_init100_AHU2_dgd.h5",
#     "20240719_043105_result_2024-07-19_end700_ssdim_net17_byz2_planrand9_dim10_init100_AHU2_dgd.h5"
# ]
# plot_distance_to_optimal(byz2_AHU2_rand9)

# byz2_AHU11e1_rand9 = [
# "20240719_143859_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bb.h5",
# "20240719_144627_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
# "20240719_145356_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
# "20240719_150128_result_2024-07-19_end400_ssdim_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5"
# ]
# plot_distance_to_optimal(byz2_AHU11e1_rand9)

# compare the different AHU (conditional numbers)
# byz_conds = [
#     "20240719_034657_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_blksz3_init100_AHU2_bb.h5",
#     "20240719_040150_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_blksz3_init100_AHU2_bcgd.h5",
#     "20240719_041627_result_2024-07-19_end700_ss1e2_net17_byz2_planrand9_dim10_init100_AHU2_dgd.h5",
#     "20240719_043105_result_2024-07-19_end700_ssdim_net17_byz2_planrand9_dim10_init100_AHU2_dgd.h5",
#     "20240719_143859_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bb.h5",
#     "20240719_144627_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     "20240719_145356_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_150128_result_2024-07-19_end400_ssdim_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5"
# ]  # it seems that the ill-conditioned case (not really, but larger AHU2), the performance is better
# plot_distance_to_optimal(byz_conds)
# TODO: try different conditional numbers under this case.



# we try for different initial values
# byz2_AHU11e1_rand9_init_10 = [
#     "20240719_152920_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init-10_AHU11e1_bb.h5",
#     "20240719_153652_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init-10_AHU11e1_bcgd.h5",
#     "20240719_154429_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_init-10_AHU11e1_dgd.h5",
#     "20240719_155204_result_2024-07-19_end400_ssdim_net17_byz2_planrand9_dim10_init-10_AHU11e1_dgd.h5"
# ]
# plot_distance_to_optimal(byz2_AHU11e1_rand9_init_10)
# still the DGD converge first, and diverge later. Different initialization seems would not change the performances.

# # Let us come back to the initialization with all 10, and try different attacks.
# byz2_AHU11e1_planxxxx9 = [
# # "20240719_143859_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bb.h5",
# # "20240719_144627_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
# # "20240719_145356_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
# # "20240719_150128_result_2024-07-19_end400_ssdim_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
# # "20240719_094216_result_2024-07-19_end400_ss1e2_net17_byz2_plantime9_dim10_blksz3_init10_AHU11e1_bb.h5",
# # "20240719_095032_result_2024-07-19_end400_ss1e2_net17_byz2_plantime9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
# # "20240719_095849_result_2024-07-19_end400_ss1e2_net17_byz2_plantime9_dim10_init10_AHU11e1_dgd.h5",
# # "20240719_100706_result_2024-07-19_end400_ssdim_net17_byz2_plantime9_dim10_init10_AHU11e1_dgd.h5",
# "20240719_085242_result_2024-07-19_end400_ss1e2_net17_byz2_planones9_dim10_init10_AHU11e1_dgd.h5",
# "20240719_090055_result_2024-07-19_end400_ssdim_net17_byz2_planones9_dim10_init10_AHU11e1_dgd.h5",
# # "20240719_090908_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_blksz3_init10_AHU11e1_bb.h5",
# # "20240719_091724_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
# # "20240719_092544_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_init10_AHU11e1_dgd.h5",
# # "20240719_093402_result_2024-07-19_end400_ssdim_net17_byz2_planflip9_dim10_init10_AHU11e1_dgd.h5"
# ]
# # The attacks result in similar performances. The commented time9 makes the dgd diverge clearly.
# plot_distance_to_optimal(byz2_AHU11e1_planxxxx9)

# byz2_init10_AHU11e1_planxxxx2 = [
#     # "20240719_184616_result_2024-07-19_end400_ssdim_net17_byz2_plantime2_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_183850_result_2024-07-19_end400_ss1e2_net17_byz2_plantime2_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_183120_result_2024-07-19_end400_ss1e2_net17_byz2_plantime2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     # "20240719_182347_result_2024-07-19_end400_ss1e2_net17_byz2_plantime2_dim10_blksz3_init10_AHU11e1_bb.h5",
#     # "20240719_181609_result_2024-07-19_end400_ssdim_net17_byz2_planflip2_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_180831_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_180052_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     # "20240719_175326_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_blksz3_init10_AHU11e1_bb.h5",
#     "20240719_174600_result_2024-07-19_end400_ssdim_net17_byz2_planrand2_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_173836_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_173109_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     "20240719_172343_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_blksz3_init10_AHU11e1_bb.h5",
#     "20240719_171619_result_2024-07-19_end400_ssdim_net17_byz2_planones2_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_170853_result_2024-07-19_end400_ss1e2_net17_byz2_planones2_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_170122_result_2024-07-19_end400_ss1e2_net17_byz2_planones2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     "20240719_165350_result_2024-07-19_end400_ss1e2_net17_byz2_planones2_dim10_blksz3_init10_AHU11e1_bb.h5"
# ]
# # all seems similar, the attack magenitude should influence a little, but anyway, larger attack magnitude is better for bb, bcgd's performance.
# # When the magnitude is 2, the performance is similar.
# plot_distance_to_optimal(byz2_init10_AHU11e1_planxxxx2)

# byz2_init10_planflip29 = [
#     "20240719_180052_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     "20240719_175326_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_blksz3_init10_AHU11e1_bb.h5",
#     # "20240719_181609_result_2024-07-19_end400_ssdim_net17_byz2_planflip2_dim10_init10_AHU11e1_dgd.h5",
#     "20240719_180831_result_2024-07-19_end400_ss1e2_net17_byz2_planflip2_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_091724_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
#     # "20240719_090908_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_blksz3_init10_AHU11e1_bb.h5",
#     "20240719_092544_result_2024-07-19_end400_ss1e2_net17_byz2_planflip9_dim10_init10_AHU11e1_dgd.h5",
#     # "20240719_093402_result_2024-07-19_end400_ssdim_net17_byz2_planflip9_dim10_init10_AHU11e1_dgd.h5"
#
# # The comparison is not clear, DGD sometiems better than BB and BCGD.
# plot_distance_to_optimal(byz2_init10_planrand29)

byz2_planrand29 = [
    "20240719_172343_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_blksz3_init10_AHU11e1_bb.h5",
    "20240719_173109_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_blksz3_init10_AHU11e1_bcgd.h5",
    "20240719_173836_result_2024-07-19_end400_ss1e2_net17_byz2_planrand2_dim10_init10_AHU11e1_dgd.h5",
    "20240719_174600_result_2024-07-19_end400_ssdim_net17_byz2_planrand2_dim10_init10_AHU11e1_dgd.h5",
    "20240719_143859_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bb.h5",
    "20240719_144627_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_blksz3_init10_AHU11e1_bcgd.h5",
    "20240719_145356_result_2024-07-19_end400_ss1e2_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
    "20240719_150128_result_2024-07-19_end400_ssdim_net17_byz2_planrand9_dim10_init10_AHU11e1_dgd.h5",
]  # DGD is similar when attack mag=2.

plot_distance_to_optimal(byz2_planrand29)

# It seems that higher AHU has better comparisons?
