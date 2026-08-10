#!/bin/bash

# ============================================================================
# SwaveNetV1 Experiments - SpikF-style Haar Wavelet Selector
# ============================================================================
#
# Model: SwaveNetV1 (Haar matrix transform + ternary spiking + SpikF selector)
# 
# Testing different keep_levels options:
#   - "all"    : Keep all Haar coefficients (approx + all detail levels)
#   - "detail" : Keep only detail coefficients (drop approximation)
#   - "approx" : Keep only approximation coefficient (low-frequency)
#   - "0,1"    : Keep approx (0) + coarsest detail (1)
#
# Datasets: ETTh1, ETTh2
# Pred lengths: 96, 336
# Seeds: 2022, 2023, 2024 (year convention)
#
# Total experiments: 2 datasets × 2 pred_lens × 3 seeds × 4 keep_levels = 48 runs
# ============================================================================

GPU=${1:-0}

echo "============================================================"
echo "SwaveNetV1 EXPERIMENTS"
echo "GPU: ${GPU}"
echo "Seeds: 2022, 2023, 2024"
echo "============================================================"

# ============================================================================
# EXPERIMENT 1: keep_levels = "all" (all coefficients)
# ============================================================================

echo ""
echo ">>> Experiment 1: keep_levels = all"
echo ""

KEEP_LEVELS="all"

for DATA in ETTh1 ETTh2; do
    
    mkdir -p Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    mkdir -p test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    
    for PRED_LEN in 96 336; do
        for SEED in 2022 2023 2024; do
            
            MODEL_NAME="SwaveNetV1_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN}_seed${SEED}"
            OUTPUT_DIR="Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            
            echo "------------------------------------------------------------"
            echo "SwaveNetV1 | ${DATA} | keep_levels=${KEEP_LEVELS} | pred=${PRED_LEN} | seed=${SEED}"
            echo "------------------------------------------------------------"
            
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
                --model SwaveNetV1 \
                --keep_levels ${KEEP_LEVELS} \
                --random_seed ${SEED} \
                --save \
                --model_name ${MODEL_NAME} \
                --checkpoints ${CHECKPOINT_DIR} \
                2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}-${KEEP_LEVELS}-seed${SEED}.txt"
        done
    done
done

# ============================================================================
# EXPERIMENT 2: keep_levels = "detail" (detail only, no approximation)
# ============================================================================

echo ""
echo ">>> Experiment 2: keep_levels = detail"
echo ""

KEEP_LEVELS="detail"

for DATA in ETTh1 ETTh2; do
    
    mkdir -p Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    mkdir -p test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    
    for PRED_LEN in 96 336; do
        for SEED in 2022 2023 2024; do
            
            MODEL_NAME="SwaveNetV1_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN}_seed${SEED}"
            OUTPUT_DIR="Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            
            echo "------------------------------------------------------------"
            echo "SwaveNetV1 | ${DATA} | keep_levels=${KEEP_LEVELS} | pred=${PRED_LEN} | seed=${SEED}"
            echo "------------------------------------------------------------"
            
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
                --model SwaveNetV1 \
                --keep_levels ${KEEP_LEVELS} \
                --random_seed ${SEED} \
                --save \
                --model_name ${MODEL_NAME} \
                --checkpoints ${CHECKPOINT_DIR} \
                2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}-${KEEP_LEVELS}-seed${SEED}.txt"
        done
    done
done

# ============================================================================
# EXPERIMENT 3: keep_levels = "approx" (approximation only)
# ============================================================================

echo ""
echo ">>> Experiment 3: keep_levels = approx"
echo ""

KEEP_LEVELS="approx"

for DATA in ETTh1 ETTh2; do
    
    mkdir -p Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    mkdir -p test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}
    
    for PRED_LEN in 96 336; do
        for SEED in 2022 2023 2024; do
            
            MODEL_NAME="SwaveNetV1_${KEEP_LEVELS}_${DATA}_pl${PRED_LEN}_seed${SEED}"
            OUTPUT_DIR="Output/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS}"
            
            echo "------------------------------------------------------------"
            echo "SwaveNetV1 | ${DATA} | keep_levels=${KEEP_LEVELS} | pred=${PRED_LEN} | seed=${SEED}"
            echo "------------------------------------------------------------"
            
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
                --model SwaveNetV1 \
                --keep_levels ${KEEP_LEVELS} \
                --random_seed ${SEED} \
                --save \
                --model_name ${MODEL_NAME} \
                --checkpoints ${CHECKPOINT_DIR} \
                2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}-${KEEP_LEVELS}-seed${SEED}.txt"
        done
    done
done

# ============================================================================
# EXPERIMENT 4: keep_levels = "0,1" (approx + coarsest detail)
# ============================================================================

echo ""
echo ">>> Experiment 4: keep_levels = 0,1 (approx + coarsest detail)"
echo ""

KEEP_LEVELS="0,1"
KEEP_LEVELS_DIR="0-1"  # For directory names (no comma)

for DATA in ETTh1 ETTh2; do
    
    mkdir -p Output/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}
    mkdir -p test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}
    
    for PRED_LEN in 96 336; do
        for SEED in 2022 2023 2024; do
            
            MODEL_NAME="SwaveNetV1_${KEEP_LEVELS_DIR}_${DATA}_pl${PRED_LEN}_seed${SEED}"
            OUTPUT_DIR="Output/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}"
            CHECKPOINT_DIR="test_results/${DATA}/SwaveNetV1_${KEEP_LEVELS_DIR}"
            
            echo "------------------------------------------------------------"
            echo "SwaveNetV1 | ${DATA} | keep_levels=${KEEP_LEVELS} | pred=${PRED_LEN} | seed=${SEED}"
            echo "------------------------------------------------------------"
            
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
                --model SwaveNetV1 \
                --keep_levels ${KEEP_LEVELS} \
                --random_seed ${SEED} \
                --save \
                --model_name ${MODEL_NAME} \
                --checkpoints ${CHECKPOINT_DIR} \
                2>&1 | tee "${OUTPUT_DIR}/pl${PRED_LEN}-${KEEP_LEVELS_DIR}-seed${SEED}.txt"
        done
    done
done

# ============================================================================
# COMPLETE
# ============================================================================

echo ""
echo "============================================================"
echo "SwaveNetV1 EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Summary of experiments:"
echo ""
echo "Model: SwaveNetV1 (Haar matrix + SpikF-style selector + ternary spiking)"
echo ""
echo "keep_levels tested:"
echo "  - all    : All Haar coefficients"
echo "  - detail : Detail only (no approximation)"
echo "  - approx : Approximation only (low-frequency)"
echo "  - 0,1    : Approx + coarsest detail"
echo ""
echo "Datasets: ETTh1, ETTh2"
echo "Pred lengths: 96, 336"
echo "Seeds: 2022, 2023, 2024"
echo ""
echo "Total runs: 48"
echo "============================================================"