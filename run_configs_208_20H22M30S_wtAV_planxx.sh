#!/bin/bash

# Config files
CONFIG_FILES=(
  test_208_20H22M30S_end100_ss1e1_blk3_wtAV_init10_flip9_net50_byz5_dim9_AHU10_b20.yml
  test_208_20H22M30S_end100_ss1e1_blk3_wtAV_init10_ones9_net50_byz5_dim9_AHU10_b20.yml
  test_208_20H22M30S_end100_ss1e1_blk3_wtAV_init10_rand9_net50_byz5_dim9_AHU10_b20.yml
  test_208_20H22M30S_end100_ss1e1_blk3_wtAV_init10_shift_net50_byz5_dim9_AHU10_b20.yml
  test_208_20H22M30S_end100_ss1e1_blk3_wtAV_init10_time1e1_net50_byz5_dim9_AHU10_b20.yml
)

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Running config file: $config_file"
    python main.py -a "$config_file"
done
