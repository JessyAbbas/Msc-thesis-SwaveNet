"""
Run best Optuna parameters on all prediction lengths.
Usage:
    python run_best_params.py --data ETTh1 --gpu 0
    
Reads best_params.json from optuna_results/{data}_pl96/ and runs
on pred_len = [96, 192, 336, 720] with 3 seeds.
"""

import os
import sys
import json
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--root_path', type=str, 
                        default='/media/homes/abbasj/SpikF/datasets/long/')
    parser.add_argument('--tuned_pl', type=int, default=96,
                        help='Prediction length that was tuned (to find best_params.json)')
    args = parser.parse_args()
    
    # Load best params
    params_file = f'optuna_results/{args.data}_pl{args.tuned_pl}/best_params.json'
    if not os.path.exists(params_file):
        print(f"ERROR: {params_file} not found. Run optuna_tuning.py first.")
        sys.exit(1)
    
    with open(params_file) as f:
        best = json.load(f)
    
    bp = best['params']
    print(f"{'='*60}")
    print(f"Running best params for {args.data}")
    print(f"{'='*60}")
    print(f"  hidden_dim: {bp['hidden_dim']}")
    print(f"  patch_dim:  {bp['patch_dim']}")
    print(f"  levels:     {bp['levels']}")
    print(f"  patch_num:  {bp['patch_num']}")
    print(f"  Tuned MSE:  {best['best_mse']:.6f}")
    print(f"{'='*60}\n")
    
    # Create output directory
    out_dir = f'optuna_results/{args.data}_best'
    os.makedirs(out_dir, exist_ok=True)
    
    pred_lens = [96, 192, 336, 720]
    
    for pl in pred_lens:
        print(f"\n--- {args.data} pred_len={pl} ---")
        
        cmd = [
            'python', 'run_long.py',
            '--data', args.data,
            '--root_path', args.root_path,
            '--data_path', f'{args.data}.csv',
            '--features', 'M',
            '--seq_len', '96',
            '--pred_len', str(pl),
            '--batch_size', '32',
            '--lr', '5e-4',
            '--levels', str(bp['levels']),
            '--patch_dim', str(bp['patch_dim']),
            '--patch_num', str(bp['patch_num']),
            '--T', '16',
            '--hidden_dim', str(bp['hidden_dim']),
            '--train_epochs', '20',
            '--patience', '5',
            '--model', 'SwaveNetV1_LightSpikformer',
            '--keep_levels', 'all',
            '--num_heads', '8',
            '--use_gray_pe', '1',
            '--use_log_pe', '1',
            '--gray_num_bits', '10',
            '--dwc_kernel', '5',
            '--save',
            '--model_name', f'best_optuna_{args.data}_pl{pl}',
            '--checkpoints', f'{out_dir}',
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        
        log_file = f'{out_dir}/pl{pl}.txt'
        print(f"  Logging to: {log_file}")
        
        with open(log_file, 'w') as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        
        if proc.returncode != 0:
            print(f"  WARNING: pred_len={pl} exited with code {proc.returncode}")
        else:
            # Parse results from log
            with open(log_file) as f:
                lines = f.readlines()
            for line in lines[-5:]:
                if 'Average MSE' in line or 'Average MAE' in line:
                    print(f"  {line.strip()}")
    
    print(f"\n{'='*60}")
    print(f"ALL DONE — Results in {out_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
