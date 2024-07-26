#!/bin/bash

# Config files
CONFIG_FILES=(
  test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init0_ones9_net50_byz1_dim9_AHU1003_b20.yml
  test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init0_rand9_net50_byz1_dim9_AHU10_b20.yml
  test_208_16H54M49S_end100_ss1e1_blk3_wtAV_init10_ones9_net50_byz1_dim9_AHU1003_b20.yml
)

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Running config file: $config_file"
    python main.py -a "$config_file"
done
