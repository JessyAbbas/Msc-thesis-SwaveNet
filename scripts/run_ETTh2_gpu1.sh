# #!/bin/bash

# # ============================================================================
# # SwaveNetV1 - ETTh2 Experiments (GPU 1)
# # ============================================================================
# # Seeds are hardcoded in run_long.py: [2022, 2023, 2024]
# # Each call runs 3 seeds internally and outputs averaged results
# # ============================================================================

# cd /media/homes/abbasj/SpikF

# echo "============================================================"
# echo "SwaveNetV1 - ETTh2 on GPU 1"
# echo "============================================================"

# DATA="ETTh2"

# for KEEP_LEVELS in all detail approx 0,1; do

#     # Handle directory naming for 0,1
#     if [ "$KEEP_LEVELS" == "0,1" ]; then
#         KEEP_LEVELS_DIR="0-1"
#     else
#         KEEP_LEVELS_DIR="$KEEP_LEVELS"
#     fi

#     mkdir -p Output/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}
#     mkdir -p test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}

#     for PRED_LEN in 96 336; do

#         MODEL_NAME="SwaveNetV1_${KEEP_LEVELS_DIR}_${DATA}_pl${PRED_LEN}"
#         OUTPUT_DIR="Output/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}"
#         CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}"

#         echo "------------------------------------------------------------"
#         echo "SwaveNetV1 | ${DATA} | keep_levels=${KEEP_LEVELS} | pred=${PRED_LEN}"
#         echo "------------------------------------------------------------"

#         CUDA_VISIBLE_DEVICES=1 python run_long.py \
#             --data ${DATA} \
#             --root_path /media/homes/abbasj/SpikF/datasets/long/ \
#             --data_path ${DATA}.csv \
#             --features M \
#             --seq_len 96 \
#             --pred_len ${PRED_LEN} \
#             --batch_size 32 \
#             --lr 5e-4 \
#             --levels 1 \
#             --patch_dim 32 \
#             --patch_num 32 \
#             --T 16 \
#             --hidden_dim 720 \
#             --train_epochs 5 \
#             --patience 3 \
#             --model SwaveNetV1 \
#             --keep_levels ${KEEP_LEVELS} \
#             --save \
#             --model_name ${MODEL_NAME} \
#             --checkpoints ${CHECKPOINT_DIR} \
#             2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}-${KEEP_LEVELS_DIR}.txt"

#     done
# done

# echo ""
# echo "============================================================"
# echo "ETTh2 COMPLETE!"
# echo "============================================================"




#!/bin/bash

# ============================================================================
# SwaveNetV1_Attention - ETTh2 Experiments (GPU 1)
# ============================================================================
# Model: SwaveNetV1 + Spike-Driven Self-Attention (SDSA)
# Seeds: hardcoded in run_long.py [2022, 2023, 2024]
# ============================================================================



#!/bin/bash

# ============================================================================
# SwaveNetV1_Attention - ETTh2 Experiments (GPU 1)
# ============================================================================
# Model: SwaveNetV1 + Spike-Driven Self-Attention (SDSA) - always enabled
# Seeds: hardcoded in run_long.py [2022, 2023, 2024]
# ============================================================================


# echo "============================================================"
# echo "SwaveNetV1_Attention - ETTh2 on GPU 1"
# echo "SDSA: always enabled"
# echo "============================================================"

# DATA="ETTh2"

# for KEEP_LEVELS in all detail; do

#     mkdir -p Output/${DATA}/SwaveNetV1_Attention_${KEEP_LEVELS}
#     mkdir -p test_results/${DATA}/SwaveNetV1_Attention_${KEEP_LEVELS}

#     for PRED_LEN in 96 336; do

#         MODEL_NAME="SwaveNetV1_Attention_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN}"
#         OUTPUT_DIR="Output/${DATA}/SwaveNetV1_Attention_${KEEP_LEVELS}"
#         CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_Attention_${KEEP_LEVELS}"

#         echo "------------------------------------------------------------"
#         echo "SwaveNetV1_Attention | ${DATA} | ${KEEP_LEVELS} | pred=${PRED_LEN}"
#         echo "------------------------------------------------------------"

#         CUDA_VISIBLE_DEVICES=1 python run_long.py \
#             --data ${DATA} \
#             --root_path /media/homes/abbasj/SpikF/datasets/long/ \
#             --data_path ${DATA}.csv \
#             --features M \
#             --seq_len 96 \
#             --pred_len ${PRED_LEN} \
#             --batch_size 32 \
#             --lr 5e-4 \
#             --levels 1 \
#             --patch_dim 32 \
#             --patch_num 32 \
#             --T 16 \
#             --hidden_dim 720 \
#             --train_epochs 5 \
#             --patience 3 \
#             --model SwaveNetV1_Attention \
#             --keep_levels ${KEEP_LEVELS} \
#             --num_heads 8 \
#             --save \
#             --model_name ${MODEL_NAME} \
#             --checkpoints ${CHECKPOINT_DIR} \
#             2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}.txt"

#     done
# done

# echo ""
# echo "============================================================"
# echo "SwaveNetV1_Attention - ETTh2 COMPLETE!"
# echo "============================================================"
# echo ""
# echo "Experiments run:"
# echo "  2 keep_levels × 2 pred_lens = 4 experiments"
# echo "============================================================"



# #!/bin/bash
# # GPU 7 — Haar + LightSpikformer — ETTh2 (all + detail)
# cd /media/homes/abbasj/SpikF
# DATA="ETTh2"

# for KEEP_LEVELS in all detail; do

#     mkdir -p Output/${DATA}/SwaveNetV1_LightSpikformer_${KEEP_LEVELS}
#     mkdir -p test_results/${DATA}/SwaveNetV1_LightSpikformer_${KEEP_LEVELS}

#     for PRED_LEN in 96 336; do

#         echo "------------------------------------------------------------"
#         echo "LightSpikformer | ${DATA} | ${KEEP_LEVELS} | pred=${PRED_LEN}"
#         echo "------------------------------------------------------------"

#         CUDA_VISIBLE_DEVICES=7 python run_long.py \
#             --data ${DATA} \
#             --root_path /media/homes/abbasj/SpikF/datasets/long/ \
#             --data_path ${DATA}.csv \
#             --features M \
#             --seq_len 96 \
#             --pred_len ${PRED_LEN} \
#             --batch_size 32 \
#             --lr 5e-4 \
#             --levels 1 \
#             --patch_dim 32 \
#             --patch_num 32 \
#             --T 16 \
#             --hidden_dim 720 \
#             --train_epochs 5 \
#             --patience 3 \
#             --model SwaveNetV1_LightSpikformer \
#             --keep_levels ${KEEP_LEVELS} \
#             --num_heads 8 \
#             --use_gray_pe 1 \
#             --use_log_pe 1 \
#             --gray_num_bits 10 \
#             --dwc_kernel 5 \
#             --save \
#             --model_name SwaveNetV1_LightSpikformer_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN} \
#             --checkpoints test_results/${DATA}/SwaveNetV1_LightSpikformer_${KEEP_LEVELS} \
#             2>&1 | tee "Output/${DATA}/SwaveNetV1_LightSpikformer_${KEEP_LEVELS}/pl${PRED_LEN}.txt"

#     done
# done

# echo "GPU 7 DONE: Haar ETTh2 x {all,detail} x {96,336} = 4"



#!/bin/bash
# ============================================================================
# ABLATION: LightSpikformer WITHOUT wavelet branch — ETTh2
# GPU 7 — pl=96 + pl=336
# ============================================================================
# SETUP: Same file swap as GPU 6 (only need to do once):
#   cp models/SwaveNetV1_LightSpikformer.py models/SwaveNetV1_LightSpikformer.py.bak
#   cp SwaveNetV1_LightSpikformer_noWavelet.py models/SwaveNetV1_LightSpikformer.py
# ============================================================================

DATA="ETTh2"

mkdir -p Output/${DATA}/SwaveNetV1_LightSpikformer_noWavelet
mkdir -p test_results/${DATA}/SwaveNetV1_LightSpikformer_noWavelet

for PRED_LEN in 96 336; do

    echo "------------------------------------------------------------"
    echo "noWavelet ablation | ${DATA} | pred=${PRED_LEN}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=7 python run_long.py \
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
        --model SwaveNetV1_LightSpikformer \
        --keep_levels all \
        --num_heads 8 \
        --use_gray_pe 1 \
        --use_log_pe 1 \
        --gray_num_bits 10 \
        --dwc_kernel 5 \
        --save \
        --model_name SwaveNetV1_LightSpikformer_noWavelet_${DATA}_pl${PRED_LEN} \
        --checkpoints test_results/${DATA}/SwaveNetV1_LightSpikformer_noWavelet \
        2>&1 | tee "Output/${DATA}/SwaveNetV1_LightSpikformer_noWavelet/pl${PRED_LEN}.txt"

done

echo "GPU 7 DONE — ETTh2 noWavelet (pl=96 + pl=336)"
# RESTORE: cp models/SwaveNetV1_LightSpikformer.py.bak models/SwaveNetV1_LightSpikformer.py