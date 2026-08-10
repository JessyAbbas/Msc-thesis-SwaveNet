#!/bin/bash

GPU=${1:-0}

echo "============================================"
echo "TEST: db8, J=2, patch_num=64, seq_len=128"
echo "GPU: ${GPU}"
echo "============================================"

cd /media/homes/abbasj/SpikF

CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
    --data ETTh1 \
    --root_path /media/homes/abbasj/SpikF/datasets/long/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 128 \
    --pred_len 96 \
    --batch_size 32 \
    --lr 5e-4 \
    --levels 1 \
    --patch_dim 32 \
    --patch_num 64 \
    --T 16 \
    --hidden_dim 720 \
    --train_epochs 1 \
    --patience 1 \
    --model SwaveNetV3 \
    --wavelet db8 \
    --dwt_level 2 \
    --scale_select both \
    --random_seed 0 \
    --gpu 0 \
    --model_name TEST_db8_pn64_seq128

echo "TEST COMPLETE"
