#!/bin/bash

# 设置 PYTHONPATH
export PYTHONPATH="/mnt/d/PhD_Qu_from_2021/Sketches/2024-07_BlockByz(code)/BlockByz07_v1"

# 定义配置文件列表
CONFIG_FILES=(
    "./configs/test_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_blksz3_init100_AHU2_bb.yml"
    "./configs/test_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_blksz3_init100_AHU2_bcgd.yml"
    "./configs/test_2024-07-19_end700_ss1e2_net17_byz2_planones9_dim10_init100_AHU2_dgd.yml"
    "./configs/test_2024-07-19_end700_ssdim_net17_byz2_planones9_dim10_init100_AHU2_dgd.yml"
)

# 检查配置文件
for config_file in "${CONFIG_FILES[@]}"; do
    echo "Checking configuration $config_file"
    python ./utils/utils.py --check-config "$config_file"
    if [ $? -ne 0 ]; then
        echo "Configuration file $config_file does not match the filename. Exiting."
        exit 1
    fi
done

# 根据配置文件名称执行算法
for config_file in "${CONFIG_FILES[@]}"; do
    filename=$(basename "$config_file")
    filename_no_ext="${filename%.*}"

    echo "------------- Processing $config_file -------------"
    echo "Filename: $filename_no_ext"

    processed=0

    if echo "$filename_no_ext" | grep -qi "_bb"; then
        echo "Running BB algorithm for $config_file"
        python ./algorithms/bb_algorithm.py -o "$config_file"
        processed=1
    fi

    if [ $processed -eq 0 ]; then
        if echo "$filename_no_ext" | grep -qi "_bcgd"; then
            echo "Running BCGD algorithm for $config_file"
            python ./algorithms/bcgd_algorithm.py -o "$config_file"
            processed=1
        fi
    fi

    if [ $processed -eq 0 ]; then
        if echo "$filename_no_ext" | grep -qi "_dgd"; then
            echo "Running DGD algorithm for $config_file"
            python ./algorithms/dgd_algorithm.py -o "$config_file"
            processed=1
        fi
    fi

    if [ $processed -eq 0 ]; then
        echo "Unknown algorithm in file $config_file. Skipping."
    fi
done

echo "--------------------------------------------------------------"

exit 0