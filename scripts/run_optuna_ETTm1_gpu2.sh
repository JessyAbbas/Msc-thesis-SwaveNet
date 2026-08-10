#!/bin/bash
# Optuna Tuning: ETTm1 on GPU 2
cd /media/homes/abbasj/SpikF
pip install optuna --break-system-packages 2>/dev/null

echo "============================================================"
echo "Optuna Tuning: ETTm1 | pred_len=96 | GPU 2"
echo "Trials: 72 | Sampler: TPE | Seeds: 2022,2023,2024"
echo "============================================================"

python optuna_tuning.py \
    --data ETTm1 \
    --pred_len 96 \
    --seq_len 96 \
    --gpu 2 \
    --n_trials 72 \
    --root_path /media/homes/abbasj/SpikF/datasets/long/ \
    2>&1 | tee "optuna_results/ETTm1_pl96/tuning_log.txt"

echo "DONE — ETTm1"
