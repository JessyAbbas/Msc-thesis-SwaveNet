#!/bin/bash
# ============================================================================
# Run ETTh1 best Optuna params — enrich_lif v_threshold=0.5
# Best params: hidden_dim=540, patch_dim=32, levels=3, patch_num=32
# ============================================================================

cd /media/homes/abbasj/SpikF

DATA="ETTh1"
GPU=4

mkdir -p Output/${DATA}/best_optuna_vth05
mkdir -p test_results/${DATA}/best_optuna_vth05

for PL in 96 192 336 720; do
    echo "============================================================"
    echo "Best Optuna vth05 | ${DATA} | pred_len=${PL} | GPU ${GPU}"
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
        --levels 3 \
        --patch_dim 32 \
        --patch_num 32 \
        --T 16 \
        --hidden_dim 540 \
        --train_epochs 20 \
        --patience 5 \
        --model SwaveNetV1_LightSpikformer_vth05 \
        --keep_levels all \
        --num_heads 8 \
        --use_gray_pe 1 \
        --use_log_pe 1 \
        --gray_num_bits 10 \
        --dwc_kernel 5 \
        --save \
        --model_name best_optuna_vth05_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/best_optuna_vth05 \
        2>&1 | tee "Output/${DATA}/best_optuna_vth05/pl${PL}.txt"
done

echo "DONE — ${DATA} vth05 on all prediction lengths"