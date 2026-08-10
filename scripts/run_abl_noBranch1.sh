#!/bin/bash
# ABLATION: noBranch1 — ETTh2 + ETTm1, pl={96,336}
cd /media/homes/abbasj/SpikF

GPU=7

# ETTh2 (hidden=720, patch_dim=32, levels=2, patch_num=16)
DATA="ETTh2"
mkdir -p Output/${DATA}/abl_noBranch1
for PL in 96 336; do
    echo "abl_noBranch1 | ${DATA} | pl=${PL}"
    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path ${DATA}.csv --features M --seq_len 96 --pred_len ${PL} \
        --batch_size 32 --lr 5e-4 --levels 2 --patch_dim 32 --patch_num 16 \
        --T 16 --hidden_dim 720 --train_epochs 20 --patience 5 \
        --model abl_noBranch1 --keep_levels all --num_heads 8 \
        --use_gray_pe 1 --use_log_pe 1 --gray_num_bits 10 --dwc_kernel 5 \
        --save --model_name abl_noBranch1_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/abl_noBranch1 \
        2>&1 | tee "Output/${DATA}/abl_noBranch1/pl${PL}.txt"
done

# ETTm1 (hidden=180, patch_dim=64, levels=2, patch_num=32)
DATA="ETTm1"
mkdir -p Output/${DATA}/abl_noBranch1
for PL in 96 336; do
    echo "abl_noBranch1 | ${DATA} | pl=${PL}"
    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path ${DATA}.csv --features M --seq_len 96 --pred_len ${PL} \
        --batch_size 32 --lr 5e-4 --levels 2 --patch_dim 64 --patch_num 32 \
        --T 16 --hidden_dim 180 --train_epochs 20 --patience 5 \
        --model abl_noBranch1 --keep_levels all --num_heads 8 \
        --use_gray_pe 1 --use_log_pe 1 --gray_num_bits 10 --dwc_kernel 5 \
        --save --model_name abl_noBranch1_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/abl_noBranch1 \
        2>&1 | tee "Output/${DATA}/abl_noBranch1/pl${PL}.txt"
done

echo "DONE — abl_noBranch1"
