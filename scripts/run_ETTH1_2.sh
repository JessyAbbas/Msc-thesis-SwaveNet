#!/bin/bash
# Experiment 1: SwaveNet Scale Selection
# Compare:
#   both:   All coefficients (your current runs)
#   detail: Only d_j coefficients (high-frequency)
# 
# Datasets: ETTh1, ETTh2

# Create output directories
mkdir -p Output/ETTh1/SwaveNet_detail
mkdir -p Output/ETTh2/SwaveNet_detail
mkdir -p test_results/ETTh1/SwaveNet_detail
mkdir -p test_results/ETTh2/SwaveNet_detail

# ETTh1 - Detail Only (d_j)
echo "Running ETTh1 - Detail Only"

for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "ETTh1, detail, pred_len=${pred_len}, seed=${seed}"
        
        CUDA_VISIBLE_DEVICES=0 python run_long.py \
            --data ETTh1 \
            --root_path /media/homes/abbasj/SpikF/datasets/long/ \
            --data_path ETTh1.csv \
            --features M \
            --seq_len 96 \
            --pred_len ${pred_len} \
            --batch_size 32 \
            --lr 5e-4 \
            --levels 1 \
            --patch_dim 32 \
            --patch_num 32 \
            --T 16 \
            --hidden_dim 720 \
            --train_epochs 5 \
            --patience 3 \
            --model_name SwaveNet_detail_ETTh1_out${pred_len}_seed${seed} \
            --model SwaveNet \
            --scale_select detail \
            --random_seed ${seed} \
            --save \
            --checkpoints test_results/ETTh1/SwaveNet_detail \
            2>&1 | tee "Output/ETTh1/SwaveNet_detail/${pred_len}-seed${seed}.txt"
    done
done

# ETTh2 - Detail Only (d_j)
echo "Running ETTh2 - Detail Only"

for seed in 0 42 79
do
    for pred_len in 96 192 336 720
    do
        echo "ETTh2, detail, pred_len=${pred_len}, seed=${seed}"
        
        CUDA_VISIBLE_DEVICES=0 python run_long.py \
            --data ETTh2 \
            --root_path /media/homes/abbasj/SpikF/datasets/long/ \
            --data_path ETTh2.csv \
            --features M \
            --seq_len 96 \
            --pred_len ${pred_len} \
            --batch_size 32 \
            --lr 5e-4 \
            --levels 1 \
            --patch_dim 32 \
            --patch_num 32 \
            --T 16 \
            --hidden_dim 720 \
            --train_epochs 5 \
            --patience 3 \
            --model_name SwaveNet_detail_ETTh2_out${pred_len}_seed${seed} \
            --model SwaveNet \
            --scale_select detail \
            --random_seed ${seed} \
            --save \
            --checkpoints test_results/ETTh2/SwaveNet_detail \
            2>&1 | tee "Output/ETTh2/SwaveNet_detail/${pred_len}-seed${seed}.txt"
    done
done


echo "Experiment 1 completed!"
echo "Compare results with your existing SwaveNet runs (scale_select=both)"