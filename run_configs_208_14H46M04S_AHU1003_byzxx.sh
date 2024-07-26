#!/bin/bash

# Config files
CONFIG_FILES=(
  test_207_16H54M46S_end100_ssdim_blk2_wtBB_init10_rand9_net20_byz1_dim4_AHU1003_b20_new.yml
  test_207_16H54M46S_end100_ssdim_blk2_wtBB_init10_rand9_net20_byz3_dim4_AHU1003_b20_new.yml
  test_207_16H54M46S_end100_ssdim_blk2_wtBB_init10_rand9_net20_byz5_dim4_AHU1003_b20_new.yml
  test_207_16H54M46S_end100_ssdim_blk2_wtBB_init10_rand9_net20_byz7_dim4_AHU1003_b20_new.yml
)

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Running config file: $config_file"
    python main.py -a "$config_file"
done