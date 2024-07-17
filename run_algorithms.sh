#!/bin/bash

export PYTHONPATH="/path/to/your/PhD_Qu_from_2021/Sketches/2024-07_BlockByz(code)/BlockByz07_v1"

# Define the configurations
CONFIG_FILES=(
    ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_blksz2_bcgd.yml
    ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_blksz2_bb.yml
    ./configs/test_base_end2000_ss1e3_net30_byz0_dim7_dgd.yml
    ./configs/test_base_end2000_ssdim_net30_byz0_dim7_dgd.yml
    ./configs/test_base_end2000_ss1e3_net30_byz1_dim7_blksz2_bcgd.yml
    ./configs/test_base_end2000_ss1e3_net30_byz1_dim7_blksz2_bb.yml
    ./configs/test_base_end2000_ss1e3_net30_byz1_dim7_dgd.yml
    ./configs/test_base_end2000_ssdim_net30_byz1_dim7_dgd.yml
)

# Check configurations
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Checking configuration $config_file"
    if ! python ./utils/utils.py --check-config "$config_file"; then
        echo "Configuration file $config_file does not match the filename. Exiting."
        exit 1
    fi
done

# Execute algorithms based on the configuration file names
for config_file in "${CONFIG_FILES[@]}"; do
    filename=$(basename "$config_file" .yml)

    echo "Processing $config_file"
    echo "Filename: $filename"

    if [[ "$filename" == *_bb* ]]; then
        echo "Running BB algorithm for $config_file"
        python ./algorithms/bb_algorithm.py -o "$config_file"
        continue
    fi

    if [[ "$filename" == *_bcgd* ]]; then
        echo "Running BCGD algorithm for $config_file"
        python ./algorithms/bcgd_algorithm.py -o "$config_file"
        continue
    fi

    if [[ "$filename" == *_dgd* ]]; then
        echo "Running DGD algorithm for $config_file"
        python ./algorithms/dgd_algorithm.py -o "$config_file"
        continue
    fi

    echo "Unknown algorithm in file $config_file. Skipping."
done

exit 0