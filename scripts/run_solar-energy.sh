for seed in 0 42 79
do
    for seq_len in 168
    do
        for pred_len in 6 24 48 96
        do
            echo solar-energy_${seq_len}-${pred_len}
            python run_long.py \
                --data solar-energy \
                --data_path solar-energy.csv \
                --features M \
                --seq_len ${seq_len} \
                --pred_len ${pred_len} \
                --batch_size 16 \
                --lr 1e-3 \
                --levels 2 \
                --patch_dim 16 \
                --patch_num 84 \
                --T 4 \
                --hidden_dim 360 \
                --train_epochs 5 \
                --patience 1 \
                --model_name SpikF_solar-energy_input${seq_len}_output${pred_len} \
                --model SpikF \
                | tee "Output/solar-energy/SpikF/${seq_len}-${pred_len}-seed${seed}.txt"
        done
    done
done