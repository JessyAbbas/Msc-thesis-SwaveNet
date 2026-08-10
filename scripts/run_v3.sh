#!/bin/bash

# Experiment 3: Float DWT (No Ternary Neurons in Transform)
# Model: SwaveNetV3
# Wavelets: haar, db4, sym4
# Datasets: ETTh1, ETTh2


GPU=${1:-2}  # Default GPU 2

echo "Experiment 3: SwaveNetV3 (Float DWT)"
echo "Using GPU: ${GPU}"


# Create directories
for DATA in ETTh1 ETTh2; do
    for WAVE in haar db4 sym4; do
        mkdir -p Output/${DATA}/SwaveNetV3_${WAVE}
        mkdir -p test_results/${DATA}/SwaveNetV3_${WAVE}
    done
done

# Run experiments
for DATA in ETTh1 ETTh2; do
    for WAVE in haar db4 sym4; do
        for PRED_LEN in 96 192 336 720; do
            for SEED in 0 42 79; do
                echo "Data: ${DATA}, Wavelet: ${WAVE}, PredLen: ${PRED_LEN}, Seed: ${SEED}"
                CUDA_VISIBLE_DEVICES=${GPU} python run_long.py \
                    --data ${DATA} \
                    --root_path /media/homes/abbasj/SpikF/datasets/long/ \
                    --data_path ${DATA}.csv \
                    --features M \
                    --seq_len 96 \
                    --pred_len ${PRED_LEN} \
                    --batch_size 32 \
                    --lr 5e-4 \
                    --levels 1 \
                    --patch_dim 32 \
                    --patch_num 32 \
                    --T 16 \
                    --hidden_dim 720 \
                    --train_epochs 5 \
                    --patience 3 \
                    --model SwaveNetV3 \
                    --wavelet ${WAVE} \
                    --scale_select both \
                    --random_seed ${SEED} \
                    --gpu 0 \
                    --save \
                    --model_name SwaveNetV3_${WAVE}_${DATA}_out${PRED_LEN}_seed${SEED} \
                    --checkpoints test_results/${DATA}/SwaveNetV3_${WAVE} \
                    2>&1 | tee "Output/${DATA}/SwaveNetV3_${WAVE}/${PRED_LEN}-seed${SEED}.txt"
            done
        done
    done
done

echo "Experiment 3 Complete!"
