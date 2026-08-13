#!/bin/bash

train_data_path="/mnt/ssd2T/datasets/clean_octosplat_data/gazebo_noisy_seg_noisy_depth/2026-06-25_13-18-05_g1_row3"
test_data_path="/mnt/ssd2T/datasets/clean_octosplat_data/gazebo_eval_data_folders/greenhouse_1/row_3"

kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
source source_temp.sh # activate conda environment
python3 scripts/octosplat.py configs/octosplat/slam.py --train_data_path $train_data_path --test_data_path $test_data_path
