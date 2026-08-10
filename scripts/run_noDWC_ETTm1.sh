#!/bin/bash
cd /media/homes/abbasj/SpikF
DATA="ETTm1"; GPU=5
mkdir -p Output/${DATA}/noDWC_final test_results/${DATA}/noDWC_final
for PL in 192 336 720; do
    echo "noDWC final | ${DATA} | pl=${PL} | GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
        --data ${DATA} --root_path /media/homes/abbasj/SpikF/datasets/long/ \
        --data_path ${DATA}.csv --features M --seq_len 96 --pred_len ${PL} \
        --batch_size 32 --lr 5e-4 --levels 2 --patch_dim 64 --patch_num 32 \
        --T 16 --hidden_dim 180 --train_epochs 20 --patience 5 \
        --model abl_noDWC --keep_levels all --num_heads 8 \
        --use_gray_pe 1 --use_log_pe 1 --gray_num_bits 10 --dwc_kernel 5 \
        --save --model_name noDWC_final_${DATA}_pl${PL} \
        --checkpoints test_results/${DATA}/noDWC_final \
        2>&1 | tee "Output/${DATA}/noDWC_final/pl${PL}.txt"
done
echo "DONE — ${DATA}"
