"""
SwaveNet Optuna Hyperparameter Tuning (Fast Version)
=====================================================
- 1 seed for tuning (seed=2024)
- 20 epochs, patience=5
- After tuning: run best params with 3 seeds separately

Usage:
    python optuna_tuning.py --data ETTh1 --gpu 0
    python optuna_tuning.py --data ETTh2 --gpu 1
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_ETT_new import Exp_ETT
from spikingjelly.clock_driven import functional


def create_args(trial, base_args):
    """Create args namespace with Optuna-suggested hyperparameters."""
    
    hidden_dim = trial.suggest_categorical('hidden_dim', [180, 360, 540, 720])
    patch_dim = trial.suggest_categorical('patch_dim', [8, 16, 32, 64])
    levels = trial.suggest_categorical('levels', [1, 2, 3])
    patch_num = trial.suggest_categorical('patch_num', [8, 16, 32])
    
    if base_args.seq_len % patch_num != 0:
        raise optuna.TrialPruned(f"seq_len {base_args.seq_len} not divisible by patch_num {patch_num}")
    
    args = argparse.Namespace(
        data=base_args.data,
        root_path=base_args.root_path,
        data_path=f'{base_args.data}.csv',
        features='M',
        target='OT',
        cols=None,
        seq_len=base_args.seq_len,
        label_len=0,
        pred_len=base_args.pred_len,
        hidden_dim=hidden_dim,
        patch_dim=patch_dim,
        levels=levels,
        patch_num=patch_num,
        T=16,
        tau=2.0,
        alpha=0.5,
        batch_size=32,
        lr=5e-4,
        train_epochs=20,
        patience=5,
        loss='mae',
        model='SwaveNetV1_LightSpikformer',
        keep_levels='all',
        num_heads=8,
        use_gray_pe=1,
        use_log_pe=1,
        gray_num_bits=10,
        dwc_kernel=5,
        num_workers=4,
        rank=base_args.gpu,
        save=False,
        model_name=f'optuna_trial_{trial.number}',
        checkpoints=f'optuna_results/{base_args.data}_pl{base_args.pred_len}/checkpoints',
        mean=0, last=0, std=0,
        scale_select='both',
        wavelet='haar', J=1,
    )
    
    return args


def objective(trial, base_args):
    """Optuna objective: train with 1 seed, return MSE."""
    
    try:
        args = create_args(trial, base_args)
    except optuna.TrialPruned:
        raise
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}/{base_args.n_trials}")
    print(f"  hidden_dim={args.hidden_dim}, patch_dim={args.patch_dim}")
    print(f"  levels={args.levels}, patch_num={args.patch_num}")
    print(f"  data={args.data}, pred_len={args.pred_len}")
    print(f"{'='*60}\n")
    
    # Single seed for tuning
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    try:
        exp = Exp_ETT(args)
        
        setting = (f'{args.model}_{args.data}_ftM_sl{args.seq_len}'
                   f'_pl{args.pred_len}_lr{args.lr}_bs{args.batch_size}'
                   f'_run0')
        
        best_model_path = exp.train(setting)
        mse, mae = exp.test(setting)
        
        print(f"\n  Trial {trial.number}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        trial.set_user_attr('mae', float(mae))
        
        del exp
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ERROR: {e}")
        raise optuna.TrialPruned(f"Training failed: {e}")
    
    return mse


def main():
    parser = argparse.ArgumentParser(description='SwaveNet Optuna Tuning')
    parser.add_argument('--data', type=str, required=True, 
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'])
    parser.add_argument('--pred_len', type=int, default=336)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=72)
    parser.add_argument('--root_path', type=str, 
                        default='/media/homes/abbasj/SpikF/datasets/long/')
    
    base_args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(base_args.gpu)
    base_args.gpu = 0
    
    out_dir = f'optuna_results/{base_args.data}_pl{base_args.pred_len}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f'{out_dir}/checkpoints', exist_ok=True)
    
    print(f"{'='*60}")
    print(f"SwaveNet Optuna Tuning (1 seed, 20 epochs)")
    print(f"{'='*60}")
    print(f"Dataset:        {base_args.data}")
    print(f"Prediction len: {base_args.pred_len}")
    print(f"Seed:           2024 (single)")
    print(f"Epochs:         20 (patience=5)")
    print(f"Trials:         {base_args.n_trials}")
    print(f"Sampler:        TPE")
    print(f"Search Space:   144 combos (72 trials = 50%)")
    print(f"{'='*60}\n")
    
    study = optuna.create_study(
        study_name=f'SwaveNet_{base_args.data}_pl{base_args.pred_len}',
        direction='minimize',
        sampler=TPESampler(seed=42),
        storage=f'sqlite:///{out_dir}/optuna_study.db',
        load_if_exists=True,
    )
    
    start_time = time.time()
    
    study.optimize(
        lambda trial: objective(trial, base_args),
        n_trials=base_args.n_trials,
        catch=(Exception,),
    )
    
    elapsed = time.time() - start_time
    
    # Results
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL: #{best.number}")
    print(f"{'='*60}")
    print(f"  MSE: {best.value:.6f}")
    print(f"  MAE: {best.user_attrs.get('mae', 'N/A')}")
    print(f"  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    
    # Save
    best_params = {
        'data': base_args.data,
        'pred_len': base_args.pred_len,
        'best_trial': best.number,
        'best_mse': float(best.value),
        'best_mae': float(best.user_attrs.get('mae', -1)),
        'params': best.params,
        'total_trials': len(study.trials),
        'elapsed_hours': elapsed / 3600,
    }
    
    with open(f'{out_dir}/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Top 5
    top = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value
    )[:5]
    
    print(f"\n{'='*60}")
    print(f"TOP 5 TRIALS")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Trial':<7} {'MSE':<12} {'MAE':<12} {'hidden':<8} {'patch_d':<8} {'levels':<7} {'patch_n':<8}")
    for rank, t in enumerate(top, 1):
        mae = t.user_attrs.get('mae', -1)
        print(f"{rank:<5} {t.number:<7} {t.value:<12.6f} {mae:<12.6f} "
              f"{t.params['hidden_dim']:<8} {t.params['patch_dim']:<8} "
              f"{t.params['levels']:<7} {t.params['patch_num']:<8}")
    
    # Print run commands for all pred_lens with 3 seeds
    bp = best.params
    print(f"\n{'='*60}")
    print(f"RUN BEST PARAMS (3 seeds, all pred_lens):")
    print(f"{'='*60}")
    for pl in [96, 192, 336, 720]:
        print(f"python run_long.py --data {base_args.data} "
              f"--root_path {base_args.root_path} "
              f"--data_path {base_args.data}.csv --features M "
              f"--seq_len 96 --pred_len {pl} --batch_size 32 --lr 5e-4 "
              f"--levels {bp['levels']} --patch_dim {bp['patch_dim']} "
              f"--patch_num {bp['patch_num']} --T 16 "
              f"--hidden_dim {bp['hidden_dim']} --train_epochs 20 --patience 5 "
              f"--model SwaveNetV1_LightSpikformer --keep_levels all "
              f"--num_heads 8 --use_gray_pe 1 --use_log_pe 1 "
              f"--gray_num_bits 10 --dwc_kernel 5 --save "
              f"--model_name best_{base_args.data}_pl{pl}\n")


if __name__ == '__main__':
    main()
