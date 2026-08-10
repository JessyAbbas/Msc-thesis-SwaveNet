#!/bin/bash
# ============================================================================
# Run ETTm1 best Optuna params on all prediction lengths (3 seeds)
# Best params: hidden_dim=180, patch_dim=64, levels=2, patch_num=32
# ============================================================================

cd /media/homes/abbasj/SpikF

DATA="ETTm1"
GPU=7

mkdir -p Output/${DATA}/best_optuna
mkdir -p test_results/${DATA}/best_optuna

for PL in 96 192 336 720; do
    echo "============================================================"
    echo "Best Optuna | ${DATA} | pred_len=${PL} | GPU ${GPU}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} \
        --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path ${DATA}.csv \
        --features M \
        --seq_len 96 \
        --pred_len ${PL} \
        --batch_size 32 \
        --lr 5e-4 \
        --levels 2 \
        --patch_dim 64 \
        --patch_num 32 \
        --T 16 \
        --hidden_dim 180 \
        --train_epochs 20 \
        --patience 5 \
        --model SwaveNetV1_LightSpikformer \
        --keep_levels all \
        --num_heads 8 \
        --use_gray_pe 1 \
        --use_log_pe 1 \
        --gray_num_bits 10 \
        --dwc_kernel 5 \
        --save \
        --model_name best_optuna_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/best_optuna \
        2>&1 | tee "Output/${DATA}/best_optuna/pl${PL}.txt"
done

echo "DONE — ${DATA} best Optuna params on all prediction lengths"
