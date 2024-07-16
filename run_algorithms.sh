#!/bin/bash

export PYTHONPATH=/home/user/Sketches/2024-07_BlockByz(code)/BlockByz07_v1

python ./algorithms/bcgd_algorithm.py -o ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_blksz2_bcgd.yml

python ./algorithms/bb_algorithm.py -o ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_blksz2_bb.yml

python ./algorithms/dgd_algorithm.py -o ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_dgd.yml

python ./algorithms/dgd_algorithm.py -o ./configs/test_base_end2000_ssdim_net30_byz0_dim7_dgd.yml


python .\algorithms\bcgd_algorithm.py -o .\configs\test_base_end2000_ss1e3_net30_byz1_dim7_blksz2_bcgd.yml

python .\algorithms\bb_algorithm.py -o .\configs\test_base_end2000_ss1e3_net30_byz1_dim7_blksz2_bb.yml

python .\algorithms\dgd_algorithm.py -o .\configs\test_base_end2000_ss1e3_net30_byz1_dim7_dgd.yml

python .\algorithms\dgd_algorithm.py -o .\configs\test_base_end2000_ssdim_net30_byz1_dim7_dgd.yml