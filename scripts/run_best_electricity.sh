#!/bin/bash
# SwaveNet best params — Electricity
cd /media/homes/abbasj/SpikF
DATA="electricity"
GPU=4
mkdir -p Output/${DATA}/best_optuna
mkdir -p test_results/${DATA}/best_optuna
for PL in 192 336 720; do
    echo "Best Optuna | ${DATA} | pl=${PL} | GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path electricity.csv --features M --seq_len 96 --pred_len ${PL} \
        --batch_size 8 --lr 5e-4 --levels 1 --patch_dim 8 --patch_num 32 \
        --T 16 --hidden_dim 360 --train_epochs 20 --patience 5 \
        --model SwaveNetV1_LightSpikformer --keep_levels all --num_heads 8 \
        --use_gray_pe 1 --use_log_pe 1 --gray_num_bits 10 --dwc_kernel 5 \
        --save --model_name best_optuna_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/best_optuna \
        2>&1 | tee "Output/${DATA}/best_optuna/pl${PL}.txt"
done
echo "DONE — ${DATA}"
