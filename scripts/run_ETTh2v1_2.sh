#!/bin/bash

# ============================================================================
# SwaveNetV1_2 - ETTh2 Experiments (GPU 6)
# ============================================================================
# Wavelets: db4, db8, sym4, sym8
# J levels: 1, 2
# keep_levels: all, detail
# Seeds: hardcoded in run_long.py [2022, 2023, 2024]
# ============================================================================



# echo "============================================================"
# echo "SwaveNetV1_2 - ETTh2 on GPU 6"
# echo "Wavelets: db4, db8, sym4, sym8"
# echo "============================================================"

# DATA="ETTh2"

# for WAVELET in db4 db8 sym4 sym8; do
#     for J in 2; do
#         for KEEP_LEVELS in all detail; do

#             mkdir -p Output/${DATA}/SwaveNetV1_2_${WAVELET}_J${J}_${KEEP_LEVELS}
#             mkdir -p test_results/${DATA}/SwaveNetV1_2_${WAVELET}_J${J}_${KEEP_LEVELS}

#             for PRED_LEN in 96 336; do

#                 MODEL_NAME="SwaveNetV1_2_${WAVELET}_J${J}_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN}"
#                 OUTPUT_DIR="Output/${DATA}/SwaveNetV1_2_${WAVELET}_J${J}_${KEEP_LEVELS}"
#                 CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_2_${WAVELET}_J${J}_${KEEP_LEVELS}"

#                 echo "------------------------------------------------------------"
#                 echo "SwaveNetV1_2 | ${DATA} | ${WAVELET} | J=${J} | ${KEEP_LEVELS} | pred=${PRED_LEN}"
#                 echo "------------------------------------------------------------"

#                 CUDA_VISIBLE_DEVICES=6 python run_long.py \
#                     --data ${DATA} \
#                     --root_path /media/homes/abbasj/SpikF/datasets/long/ \
#                     --data_path ${DATA}.csv \
#                     --features M \
#                     --seq_len 96 \
#                     --pred_len ${PRED_LEN} \
#                     --batch_size 32 \
#                     --lr 5e-4 \
#                     --levels 1 \
#                     --patch_dim 32 \
#                     --patch_num 32 \
#                     --T 16 \
#                     --hidden_dim 720 \
#                     --train_epochs 5 \
#                     --patience 3 \
#                     --model SwaveNetV1_2 \
#                     --wavelet ${WAVELET} \
#                     --J ${J} \
#                     --keep_levels ${KEEP_LEVELS} \
#                     --save \
#                     --model_name ${MODEL_NAME} \
#                     --checkpoints ${CHECKPOINT_DIR} \
#                     2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}.txt"

#             done
#         done
#     done
# done

# echo ""
# echo "============================================================"
# echo "SwaveNetV1_2 - ETTh2 COMPLETE!"
# echo "============================================================"
# echo ""
# echo "Experiments run:"
# echo "  4 wavelets × 2 J levels × 2 keep_levels × 2 pred_lens = 32 experiments"
# echo "============================================================"


#!/bin/bash
# GPU 7 — db4 + LightSpikformer — ETTh2 (all, detail)

DATA="ETTh2"

for KEEP_LEVELS in all detail; do
    mkdir -p Output/${DATA}/V1_2_LightSpik_db4_${KEEP_LEVELS}
    mkdir -p test_results/${DATA}/V1_2_LightSpik_db4_${KEEP_LEVELS}

    for PRED_LEN in 96 336; do
        echo "--- db4+LightSpik | ${DATA} | ${KEEP_LEVELS} | pl=${PRED_LEN} ---"

        CUDA_VISIBLE_DEVICES=7 python run_long.py \
            --data ${DATA} \
            --root_path /media/homes/abbasj/SpikF/datasets/long/ \
            --data_path ${DATA}.csv \
            --features M --seq_len 96 --pred_len ${PRED_LEN} \
            --batch_size 32 --lr 5e-4 --levels 1 \
            --patch_dim 32 --patch_num 32 --T 16 \
            --hidden_dim 720 --train_epochs 5 --patience 3 \
            --model SwaveNetV1_2_LightSpikformer \
            --wavelet db4 --J 5 \
            --keep_levels ${KEEP_LEVELS} \
            --num_heads 8 --use_gray_pe 1 --use_log_pe 1 \
            --gray_num_bits 10 --dwc_kernel 5 \
            --save \
            --model_name V1_2_LightSpik_db4_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN} \
            --checkpoints test_results/${DATA}/V1_2_LightSpik_db4_${KEEP_LEVELS} \
            2>&1 | tee "Output/${DATA}/V1_2_LightSpik_db4_${KEEP_LEVELS}/pl${PRED_LEN}.txt"
    done
done

echo "GPU 7 DONE: db4 ETTh2 x {all,detail} x {96,336} = 4"