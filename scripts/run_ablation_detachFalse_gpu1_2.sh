#!/bin/bash
# Ablation: detachFalse — ETTh1 (GPU 1) then ETTh2 (GPU 2)
cd /media/homes/abbasj/SpikF
for DATA in ETTh1 ETTh2; do
    [ "$DATA" == "ETTh1" ] && GPU=1 || GPU=2
    mkdir -p Output/${DATA}/ablation_detachFalse
    mkdir -p test_results/${DATA}/ablation_detachFalse
    for PRED_LEN in 96 336; do
        echo "ablation_detachFalse | ${DATA} | pred=${PRED_LEN} | GPU ${GPU}"
        CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
            --data ${DATA} --root_path /media/homes/abbasj/SpikF/datasets/long/ \
            --data_path ${DATA}.csv --features M --seq_len 96 --pred_len ${PRED_LEN} \
            --batch_size 32 --lr 5e-4 --levels 1 --patch_dim 32 --patch_num 32 \
            --T 16 --hidden_dim 720 --train_epochs 5 --patience 3 \
            --model ablation_detachFalse --keep_levels all --num_heads 8 \
            --use_gray_pe 1 --use_log_pe 1 --gray_num_bits 10 --dwc_kernel 5 \
            --save --model_name ablation_detachFalse_${DATA}_pl${PRED_LEN} \
            --checkpoints test_results/${DATA}/ablation_detachFalse \
            2>&1 | tee "Output/${DATA}/ablation_detachFalse/pl${PRED_LEN}.txt"
    done
done
echo "DONE — ablation_detachFalse"
