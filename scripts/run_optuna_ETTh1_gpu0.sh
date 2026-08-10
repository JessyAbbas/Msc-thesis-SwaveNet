#!/bin/bash
# Optuna Tuning: ETTh1 on GPU 0
cd /media/homes/abbasj/SpikF
pip install optuna --break-system-packages 2>/dev/null

echo "============================================================"
echo "Optuna Tuning: ETTh1 | pred_len=96 | GPU 0"
echo "Trials: 72 | Sampler: TPE | Seeds: 2022,2023,2024"
echo "============================================================"

python optuna_tuning.py \
    --data ETTh1 \
    --pred_len 96 \
    --seq_len 96 \
    --gpu 0 \
    --n_trials 72 \
    --root_path /media/homes/abbasj/SpikF/datasets/long/ \
    2>&1 | tee "optuna_results/ETTh1_pl96/tuning_log.txt"

echo "DONE — ETTh1"
