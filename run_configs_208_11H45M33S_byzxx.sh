#!/bin/bash

# Config files
CONFIG_FILES=(
  test_208_11H45M33S_end100_ss1e2_blk1_wtBB_init10_rand9_net50_byz2_dim9_AHU100_b20.yml
  test_208_11H45M33S_end100_ss1e2_blk1_wtBB_init10_rand9_net50_byz5_dim9_AHU100_b20.yml
  test_208_11H45M33S_end100_ss1e2_blk1_wtBB_init10_rand9_net50_byz7_dim9_AHU100_b20.yml
)

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Running config file: $config_file"
    python main.py -a "$config_file"
done