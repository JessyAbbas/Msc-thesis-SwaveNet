#!/bin/bash
# SpikF baseline — electricity (epochs=20, patience=5, seeds=2022,2023,2024)
cd /media/homes/abbasj/SpikF

DATA="electricity"
GPU=5

mkdir -p Output/${DATA}/SpikF_baseline
mkdir -p test_results/${DATA}/SpikF_baseline

for PL in 720; do
    echo "============================================================"
    echo "SpikF Baseline | ${DATA} | pred_len=${PL} | GPU ${GPU}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} \
        --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path electricity.csv \
        --features M \
        --seq_len 96 \
        --pred_len ${PL} \
        --batch_size 8 \
        --lr 5e-4 \
        --levels 1 \
        --patch_dim 32 \
        --patch_num 32 \
        --T 16 \
        --hidden_dim 720 \
        --train_epochs 20 \
        --patience 5 \
        --model SpikF \
        --save \
        --model_name SpikF_baseline_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/SpikF_baseline \
        2>&1 | tee "Output/${DATA}/SpikF_baseline/pl${PL}.txt"
done

echo "DONE — SpikF baseline electricity"
