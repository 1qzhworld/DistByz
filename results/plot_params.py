from utils.utils import plot_distance_to_optimal

byz0_filenames = [
    "20240717_173944_result_base_end1000_ss1e3_net30_byz0_dim7_blksz2_bcgd.h5",
    "20240717_175918_result_base_end1000_ss1e3_net30_byz0_dim7_blksz2_bb.h5",
    "20240717_181905_result_base_end1000_ss1e3_net30_byz0_dim7_dgd.h5",
    "20240717_183838_result_base_end1000_ssdim_net30_byz0_dim7_dgd.h5"
]

plot_distance_to_optimal(byz0_filenames)


byz1_filenames = [
    "20240717_185808_result_base_end1000_ss1e3_net30_byz1_dim7_blksz2_bcgd.h5",
    "20240717_191801_result_base_end1000_ss1e3_net30_byz1_dim7_blksz2_bb.h5",
    "20240717_193816_result_base_end1000_ss1e3_net30_byz1_dim7_dgd.h5",
    # "20240717_195823_result_base_end1000_ssdim_net30_byz1_dim7_dgd.h5"
]

plot_distance_to_optimal(byz1_filenames)
