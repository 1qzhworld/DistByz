#!/bin/bash

# Config files
CONFIG_FILES=(
  test_209_10H28M31S_blk2_wtAV_init10_ones20_byz5_N50L10D9ss1e1.yml
  test_209_10H28M31S_blk2_wtAV_init10_ones100_byz5_N50L10D9ss1e1.yml
  test_209_10H28M31S_blk2_wtAV_init10_ones100_byz5_N50L10D9ss5e2.yml
  test_209_10H28M31S_blk3_wtAV_init10_ones9_byz5_N50L10D9ss1e1.yml
)

# Loop through all config files
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Running config file: $config_file"
    python main.py -a "$config_file"
done
