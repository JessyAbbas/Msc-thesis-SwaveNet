GPUS=${GPUS:-"6"}        # override with e.g. GPUS="1" or GPUS="2 3"
gpus=($GPUS)
gpu=${gpus[0]}
for seed in 0 42 79
do
    for seq_len in 96
    do
        for pred_len in 96 192 336 720
        do
            echo "ETTh2_${seq_len}-${pred_len}_seed${seed} -> physical GPU ${gpu}"
            mkdir -p "Output/ETTh2/SwaveNet/"    
            mkdir -p "test_results/ETTh2/SwaveNet/"  
            CUDA_VISIBLE_DEVICES=${gpu} python run_long.py \
                --data ETTh2 \
                --root_path /media/homes/abbasj/SpikF/datasets/long/ \
                --data_path ETTh2.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --batch_size 32 \
                --lr 5e-4 \
                --levels 1 \
                --patch_dim 32 \
                --patch_num 32 \
                --T 16 \
                --hidden_dim 720 \
                --train_epochs 5 \
                --patience 2 \
                --model_name SwaveNet_ETTh2_input${seq_len}_output${pred_len} \
                --model SwaveNet \
                --random_seed ${seed} \
                --gpu 0 \
                --save \
                --checkpoints test_results/ETTh2/SwaveNet \
                2>&1 | tee "Output/ETTh2/SwaveNet/${seq_len}-${pred_len}-seed${seed}.txt"
        done
    done
done

# Changed model name to SwaveNet
# Changed folder name to from SpikF SwaveNet
# Changed folder name to from SpikF SwaveNet