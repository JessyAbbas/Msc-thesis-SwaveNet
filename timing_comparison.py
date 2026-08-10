"""
Inference Timing Comparison: SwaveNet vs SpikF
================================================
Measures inference time and throughput on test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python timing_comparison.py --data ETTh1 --gpu 0
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_ETT_new import Exp_ETT


def create_args(data, pred_len, model, gpu=0):
    """Create args matching best Optuna params per dataset."""
    
    BEST_PARAMS = {
        'ETTh1': {'hidden_dim': 540, 'patch_dim': 32, 'levels': 3, 'patch_num': 32},
        'ETTh2': {'hidden_dim': 720, 'patch_dim': 32, 'levels': 2, 'patch_num': 16},
        'ETTm1': {'hidden_dim': 180, 'patch_dim': 64, 'levels': 2, 'patch_num': 32},
        'ETTm2': {'hidden_dim': 180, 'patch_dim': 64, 'levels': 3, 'patch_num': 8},
        'exchange': {'hidden_dim': 180, 'patch_dim': 64, 'levels': 1, 'patch_num': 32},
        'weather': {'hidden_dim': 720, 'patch_dim': 64, 'levels': 3, 'patch_num': 8},
    }
    
    DATA_PATH = {
        'ETTh1': 'ETTh1.csv', 'ETTh2': 'ETTh2.csv',
        'ETTm1': 'ETTm1.csv', 'ETTm2': 'ETTm2.csv',
        'exchange': 'exchange_rate.csv', 'weather': 'weather.csv',
    }
    
    # SpikF uses fixed params
    if model == 'SpikF':
        params = {'hidden_dim': 720, 'patch_dim': 32, 'levels': 1, 'patch_num': 32}
    else:
        params = BEST_PARAMS[data]
    
    args = argparse.Namespace(
        data=data,
        root_path='/media/homes/abbasj/SpikF/datasets/long/',
        data_path=DATA_PATH[data],
        features='M', target='OT', cols=None,
        seq_len=96, label_len=0, pred_len=pred_len,
        hidden_dim=params['hidden_dim'],
        patch_dim=params['patch_dim'],
        levels=params['levels'],
        patch_num=params['patch_num'],
        T=16, tau=2.0, alpha=0.5,
        batch_size=32, lr=5e-4,
        train_epochs=0, patience=5, loss='mae',
        model=model,
        keep_levels='all', num_heads=8,
        use_gray_pe=1, use_log_pe=1,
        gray_num_bits=10, dwc_kernel=5,
        num_workers=4, rank=gpu,
        save=False, model_name='timing_test',
        checkpoints='timing_checkpoints',
        mean=0, last=0, std=0,
        scale_select='both',
        wavelet='haar', J=1,
    )
    return args


def measure_inference(data, pred_len, model, gpu, num_warmup=5, num_runs=20):
    """Measure inference time on test set."""
    
    args = create_args(data, pred_len, model, gpu)
    exp = Exp_ETT(args)
    
    # Build test dataloader
    setting = f'{model}_{data}_timing_pl{pred_len}'
    test_loader = exp._get_data(flag='test')
    #test_data, test_loader = exp._get_data(flag='test')
    
    # Get one batch for parameter counting
    batch_x, batch_y = next(iter(test_loader))
    batch_x = batch_x.float().to(exp.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in exp.model.parameters())
    trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    
    # Warmup runs
    exp.model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            from spikingjelly.clock_driven import functional
            functional.reset_net(exp.model)
            _ = exp.model(batch_x)
    
    # Timed runs - single batch
    batch_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            functional.reset_net(exp.model)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = exp.model(batch_x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            batch_times.append(end - start)
    
    # Timed run - full test set
    torch.cuda.synchronize()
    full_start = time.perf_counter()
    num_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(exp.device)
            functional.reset_net(exp.model)
            _ = exp.model(batch_x)
            num_samples += batch_x.shape[0]
    torch.cuda.synchronize()
    full_elapsed = time.perf_counter() - full_start
    
    results = {
        'model': model,
        'data': data,
        'pred_len': pred_len,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'batch_size': args.batch_size,
        'batch_time_mean_ms': np.mean(batch_times) * 1000,
        'batch_time_std_ms': np.std(batch_times) * 1000,
        'full_test_time_s': full_elapsed,
        'num_test_samples': num_samples,
        'throughput_samples_per_sec': num_samples / full_elapsed,
    }
    
    del exp
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange', 'weather'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[96, 336])
    base_args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(base_args.gpu)
    
    os.makedirs('timing_results', exist_ok=True)
    
    all_results = []
    
    for model in ['abl_noDWC', 'SpikF']:
        for pl in base_args.pred_lens:
            print(f"\n{'='*60}")
            print(f"Timing: {model} | {base_args.data} | pl={pl}")
            print(f"{'='*60}")
            
            try:
                r = measure_inference(base_args.data, pl, model, 0)
                all_results.append(r)
                
                print(f"  Params: {r['total_params']:,}")
                print(f"  Batch time: {r['batch_time_mean_ms']:.2f} ± {r['batch_time_std_ms']:.2f} ms")
                print(f"  Full test: {r['full_test_time_s']:.2f}s ({r['num_test_samples']} samples)")
                print(f"  Throughput: {r['throughput_samples_per_sec']:.1f} samples/sec")
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"TIMING COMPARISON — {base_args.data}")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'PL':<6} {'Params':>10} {'Batch(ms)':>12} {'Throughput':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<25} {r['pred_len']:<6} {r['total_params']:>10,} "
              f"{r['batch_time_mean_ms']:>9.2f}ms {r['throughput_samples_per_sec']:>9.1f}/s")
    
    # Save results
    outfile = f"timing_results/{base_args.data}_timing.json"
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == '__main__':
    main()
