for seed in 0 42 79
do
    for seq_len in 96
    do
        for pred_len in 96 192 336 720
        do
            echo "ETTh1_${seq_len}-${pred_len}_seed${seed}"
            mkdir -p "Output/ETTh1/SwaveNet/"
            mkdir -p "test_results/ETTh1/SwaveNet/" 
            CUDA_VISIBLE_DEVICES=0 python run_long.py \
                --data ETTh1 \
                --root_path /media/homes/abbasj/SpikF/datasets/long/ \
                --data_path ETTh1.csv \
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
                --patience 3 \
                --model_name SwaveNet_ETTh1_input${seq_len}_output${pred_len}_seed${seed} \
                --model SwaveNet \
                --random_seed ${seed} \
                --gpu 0 \
                --save \
                --checkpoints test_results/ETTh1/SwaveNet \
                2>&1 | tee "Output/ETTh1/SwaveNet/${seq_len}-${pred_len}-seed${seed}.txt"
        done
    done
done