"""
SwaveNet Energy Calculation
=============================
Computes theoretical energy consumption following:
1. Simple SOPs approach (SpikF Table 2): SOPs = alpha * T * FLOPs
2. Detailed approach (Lemaire et al., SpikF Table 3): E_Mem + E_Opts + E_Addr

Usage:
    CUDA_VISIBLE_DEVICES=0 python energy_calculation.py --data ETTh2 --pred_len 96
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_ETT_new import Exp_ETT
from spikingjelly.clock_driven import functional
from calculate import (
    fnn_snn, fnn_ann,
    addr_fnn_snn, addr_fnn_ann,
    mem_fnn_snn, mem_fnn_ann,
    sfft_sops, sifft_sops, fft, ifft,
    PJ_PER_MAC, PJ_PER_ADD,
)


# ============================================================================
# Energy constants (45nm CMOS technology)
# ============================================================================
E_MAC = 4.6   # pJ per MAC (ANN floating-point multiply-accumulate)
E_AC  = 0.9   # pJ per AC  (SNN spike accumulate)


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

# Dataset dimensions
DATA_DIM = {
    'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
    'exchange': 8, 'weather': 21,
}


def create_args(data, pred_len, model, gpu=0):
    params = BEST_PARAMS[data]
    if model == 'SpikF':
        params = {'hidden_dim': 720, 'patch_dim': 32, 'levels': 1, 'patch_num': 32}
    return argparse.Namespace(
        data=data, root_path='/media/homes/abbasj/SpikF/datasets/long/',
        data_path=DATA_PATH[data], features='M', target='OT', cols=None,
        seq_len=96, label_len=0, pred_len=pred_len,
        hidden_dim=params['hidden_dim'], patch_dim=params['patch_dim'],
        levels=params['levels'], patch_num=params['patch_num'],
        T=16, tau=2.0, alpha=0.5, batch_size=32, lr=5e-4,
        train_epochs=0, patience=5, loss='mae', model=model,
        keep_levels='all', num_heads=8, use_gray_pe=1, use_log_pe=1,
        gray_num_bits=10, dwc_kernel=5, num_workers=4, rank=gpu,
        save=False, model_name='energy_test', checkpoints='energy_ckpt',
        mean=0, last=0, std=0, scale_select='both', wavelet='haar', J=1,
    )


# ============================================================================
# 1. MEASURE FIRING RATES
# ============================================================================
def measure_firing_rates(model, test_loader, device, num_batches=20):
    """Measure per-layer firing rates during inference."""
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    
    firing_data = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                # Binary {0,1}: rate = mean
                # Ternary {-1,0,+1}: rate = abs().mean()
                rate = out.abs().float().mean().item()
                total_spikes = out.abs().float().sum().item()
                total_elements = out.numel()
                if name not in firing_data:
                    firing_data[name] = {'rates': [], 'spikes': [], 'elements': []}
                firing_data[name]['rates'].append(rate)
                firing_data[name]['spikes'].append(total_spikes)
                firing_data[name]['elements'].append(total_elements)
        return hook_fn
    
    for name, module in model.named_modules():
        if isinstance(module, MultiStepLIFNode):
            hooks.append(module.register_forward_hook(make_hook(name)))
        if 'NegIF' in type(module).__name__ or 'Ternary' in type(module).__name__:
            hooks.append(module.register_forward_hook(make_hook(f'{name}(ternary)')))
    
    model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch_data in test_loader:
            if batch_count >= num_batches:
                break
            if isinstance(batch_data, (list, tuple)):
                batch_x = batch_data[0].float().to(device)
            else:
                batch_x = batch_data.float().to(device)
            functional.reset_net(model)
            _ = model(batch_x)
            batch_count += 1
    
    for h in hooks:
        h.remove()
    
    # Compute averages
    results = {}
    for name, data in firing_data.items():
        results[name] = {
            'avg_rate': float(np.mean(data['rates'])),
            'avg_spikes': float(np.mean(data['spikes'])),
            'avg_elements': float(np.mean(data['elements'])),
        }
    
    overall_rate = np.mean([r['avg_rate'] for r in results.values()])
    
    return results, float(overall_rate)


# ============================================================================
# 2. SIMPLE SOPs ENERGY (SpikF Table 2 approach)
# ============================================================================
def compute_simple_energy(model, firing_rate, T, data, pred_len):
    """
    Simple SOPs-based energy following SpikF Eq 42:
        SOPs(l) = alpha * T * FLOPs(l)
        E_SNN = SOPs * E_AC
        E_ANN = FLOPs * E_MAC
    """
    total_flops = 0
    layer_details = []
    
    for name, module in model.named_modules():
        flops = 0
        layer_type = type(module).__name__
        
        if isinstance(module, torch.nn.Linear):
            # FLOPs = N_in * N_out (per application)
            N_in = module.in_features
            N_out = module.out_features
            flops = N_in * N_out
            if module.bias is not None:
                flops += N_out
            layer_details.append({
                'name': name, 'type': 'Linear',
                'flops': flops, 'N_in': N_in, 'N_out': N_out
            })
        
        elif isinstance(module, torch.nn.Conv1d):
            N_in = module.in_channels
            N_out = module.out_channels
            K = module.kernel_size[0]
            groups = module.groups
            # Approximate output length
            flops = N_out * (N_in // groups) * K
            if module.bias is not None:
                flops += N_out
            layer_details.append({
                'name': name, 'type': 'Conv1d',
                'flops': flops, 'channels': f'{N_in}->{N_out}', 'kernel': K
            })
        
        if flops > 0:
            total_flops += flops
    
    # SOPs for SNN
    total_sops = firing_rate * T * total_flops
    
    # Energy
    E_ann_uj = total_flops * E_MAC * 1e-6  # µJ
    E_snn_uj = total_sops * E_AC * 1e-6    # µJ
    
    ratio = E_ann_uj / E_snn_uj if E_snn_uj > 0 else float('inf')
    
    return {
        'total_flops': total_flops,
        'total_sops': total_sops,
        'firing_rate': firing_rate,
        'T': T,
        'E_ann_uj': E_ann_uj,
        'E_snn_uj': E_snn_uj,
        'ratio_ann_over_snn': ratio,
        'layer_details': layer_details,
    }


# ============================================================================
# 3. DETAILED ENERGY (Lemaire et al. approach using calculate.py)
# ============================================================================
def compute_detailed_energy(model, firing_rates_per_layer, T, data, pred_len):
    """
    Detailed energy using supervisor's calculate.py functions.
    Computes E_Opts + E_Mem + E_Addr per layer.
    """
    D = DATA_DIM[data]
    params = BEST_PARAMS[data]
    N = params['patch_num']
    p_d = params['patch_dim']
    h_dim = params['hidden_dim']
    H = pred_len
    levels = params['levels']
    
    avg_rate = np.mean([v['avg_rate'] for v in firing_rates_per_layer.values()])
    
    total_ops_uj = 0.0
    total_addr_uj = 0.0
    total_mem_uj = 0.0
    
    layer_energies = []
    
    # --- SPE: Linear(seq_len/N -> p_d) ---
    spe_in = 96 // N  # patch length
    spe_out = p_d
    spe_spikes_in = int(avg_rate * T * spe_in * D)
    spe_spikes_out = int(avg_rate * T * spe_out * D)
    
    e_ops = fnn_snn(T=T, N_out=spe_out, position_1=N, position_2=D,
                    input_spikes=spe_spikes_in, output_spikes=spe_spikes_out,
                    bias=True, lif_leak=True)
    e_addr = addr_fnn_snn(N_out=spe_out, input_events=spe_spikes_in)
    e_mem = mem_fnn_snn(T=T, N_in=spe_in, N_out=spe_out,
                        position_1=N, position_2=D,
                        input_events=spe_spikes_in, output_events=spe_spikes_out,
                        bias=True, lif_leak=True)
    
    spe_total = e_ops + e_addr + e_mem['total_energy_uj']
    total_ops_uj += e_ops
    total_addr_uj += e_addr
    total_mem_uj += e_mem['total_energy_uj']
    layer_energies.append({'layer': 'SPE', 'ops': e_ops, 'addr': e_addr,
                          'mem': e_mem['total_energy_uj'], 'total': spe_total})
    
    # --- Per block (repeated 'levels' times) ---
    for blk in range(levels):
        # Haar DWT: equivalent to Linear(N -> N) — matrix multiply W @ x
        dwt_in = N
        dwt_out = N
        dwt_spikes_in = int(avg_rate * T * dwt_in * p_d * D)
        dwt_spikes_out = int(avg_rate * T * dwt_out * p_d * D)
        
        e_ops = fnn_snn(T=T, N_out=dwt_out, position_1=p_d, position_2=D,
                        input_spikes=dwt_spikes_in, output_spikes=dwt_spikes_out,
                        bias=False, lif_leak=False)
        e_addr = addr_fnn_snn(N_out=dwt_out, input_events=dwt_spikes_in)
        e_mem = mem_fnn_snn(T=T, N_in=dwt_in, N_out=dwt_out,
                            position_1=p_d, position_2=D,
                            input_events=dwt_spikes_in, output_events=dwt_spikes_out,
                            bias=False, lif_leak=False, stateful_neuron=True)
        
        dwt_total = e_ops + e_addr + e_mem['total_energy_uj']
        total_ops_uj += e_ops
        total_addr_uj += e_addr
        total_mem_uj += e_mem['total_energy_uj']
        layer_energies.append({'layer': f'Block{blk}_S-DWT', 'ops': e_ops,
                              'addr': e_addr, 'mem': e_mem['total_energy_uj'],
                              'total': dwt_total})
        
        # SWS Selector: Linear(N -> N)
        sel_spikes_in = int(avg_rate * T * N * p_d * D)
        sel_spikes_out = int(avg_rate * T * N * p_d * D)
        
        e_ops = fnn_snn(T=T, N_out=N, position_1=p_d, position_2=D,
                        input_spikes=sel_spikes_in, output_spikes=sel_spikes_out,
                        bias=True, lif_leak=True)
        e_addr = addr_fnn_snn(N_out=N, input_events=sel_spikes_in)
        e_mem = mem_fnn_snn(T=T, N_in=N, N_out=N,
                            position_1=p_d, position_2=D,
                            input_events=sel_spikes_in, output_events=sel_spikes_out,
                            bias=True, lif_leak=True)
        
        sel_total = e_ops + e_addr + e_mem['total_energy_uj']
        total_ops_uj += e_ops
        total_addr_uj += e_addr
        total_mem_uj += e_mem['total_energy_uj']
        layer_energies.append({'layer': f'Block{blk}_SWS_Selector', 'ops': e_ops,
                              'addr': e_addr, 'mem': e_mem['total_energy_uj'],
                              'total': sel_total})
        
        # Haar iDWT: equivalent to Linear(N -> N)
        e_ops_idwt = fnn_snn(T=T, N_out=N, position_1=p_d, position_2=D,
                             input_spikes=dwt_spikes_in, output_spikes=dwt_spikes_out,
                             bias=False, lif_leak=False)
        e_addr_idwt = addr_fnn_snn(N_out=N, input_events=dwt_spikes_in)
        e_mem_idwt = mem_fnn_snn(T=T, N_in=N, N_out=N,
                                 position_1=p_d, position_2=D,
                                 input_events=dwt_spikes_in, output_events=dwt_spikes_out,
                                 bias=False, lif_leak=False, stateful_neuron=True)
        
        idwt_total = e_ops_idwt + e_addr_idwt + e_mem_idwt['total_energy_uj']
        total_ops_uj += e_ops_idwt
        total_addr_uj += e_addr_idwt
        total_mem_uj += e_mem_idwt['total_energy_uj']
        layer_energies.append({'layer': f'Block{blk}_S-iDWT', 'ops': e_ops_idwt,
                              'addr': e_addr_idwt, 'mem': e_mem_idwt['total_energy_uj'],
                              'total': idwt_total})
        
        # WGM: Linear(N -> 1)
        wgm_spikes_in = int(avg_rate * T * N * p_d * D)
        wgm_spikes_out = int(avg_rate * T * 1 * p_d * D)
        
        e_ops = fnn_snn(T=T, N_out=1, position_1=p_d, position_2=D,
                        input_spikes=wgm_spikes_in, output_spikes=wgm_spikes_out,
                        bias=False, lif_leak=True)
        e_addr = addr_fnn_snn(N_out=1, input_events=wgm_spikes_in)
        e_mem = mem_fnn_snn(T=T, N_in=N, N_out=1,
                            position_1=p_d, position_2=D,
                            input_events=wgm_spikes_in, output_events=wgm_spikes_out,
                            bias=False, lif_leak=True)
        
        wgm_total = e_ops + e_addr + e_mem['total_energy_uj']
        total_ops_uj += e_ops
        total_addr_uj += e_addr
        total_mem_uj += e_mem['total_energy_uj']
        layer_energies.append({'layer': f'Block{blk}_WGM', 'ops': e_ops,
                              'addr': e_addr, 'mem': e_mem['total_energy_uj'],
                              'total': wgm_total})
        
        # XNOR-SSA: Q,K,V projections (3 x Linear(p_d -> p_d))
        # + attention (binary matmul = ACs only)
        # + output projection Linear(p_d -> p_d)
        num_heads = 8
        d_h = p_d // num_heads
        
        for proj_name in ['Q', 'K', 'V', 'Out']:
            proj_in = p_d
            proj_out = p_d
            proj_spikes_in = int(avg_rate * T * proj_in * N * D)
            proj_spikes_out = int(avg_rate * T * proj_out * N * D)
            
            e_ops = fnn_snn(T=T, N_out=proj_out, position_1=N, position_2=D,
                            input_spikes=proj_spikes_in, output_spikes=proj_spikes_out,
                            bias=True, lif_leak=True)
            e_addr = addr_fnn_snn(N_out=proj_out, input_events=proj_spikes_in)
            e_mem = mem_fnn_snn(T=T, N_in=proj_in, N_out=proj_out,
                                position_1=N, position_2=D,
                                input_events=proj_spikes_in, output_events=proj_spikes_out,
                                bias=True, lif_leak=True)
            
            proj_total = e_ops + e_addr + e_mem['total_energy_uj']
            total_ops_uj += e_ops
            total_addr_uj += e_addr
            total_mem_uj += e_mem['total_energy_uj']
            layer_energies.append({'layer': f'Block{blk}_SSA_{proj_name}', 'ops': e_ops,
                                  'addr': e_addr, 'mem': e_mem['total_energy_uj'],
                                  'total': proj_total})
        
        # XNOR attention map: Q @ K^T — binary matmul (ACs only)
        # For each head: L x d_h @ d_h x L = L x L
        # Total ACs = num_heads * L * L * d_h * avg_rate * T
        attn_acs = int(num_heads * N * N * (d_h + 10) * avg_rate * T * D)  # +10 for Gray-PE bits
        attn_energy_uj = attn_acs * PJ_PER_ADD * 1e-6
        total_ops_uj += attn_energy_uj
        layer_energies.append({'layer': f'Block{blk}_XNOR_Attn', 'ops': attn_energy_uj,
                              'addr': 0, 'mem': 0, 'total': attn_energy_uj})
    
    # --- Decoder: Linear(N*p_d -> h_dim) + Linear(h_dim -> H) ---
    # Decoder is ANN (no spikes), so uses MACs
    dec1_flops = N * p_d * h_dim
    dec2_flops = h_dim * H
    dec_energy_uj = (dec1_flops + dec2_flops) * PJ_PER_MAC * 1e-6
    total_ops_uj += dec_energy_uj
    layer_energies.append({'layer': 'Decoder', 'ops': dec_energy_uj,
                          'addr': 0, 'mem': 0, 'total': dec_energy_uj})
    
    grand_total = total_ops_uj + total_addr_uj + total_mem_uj
    
    return {
        'E_Opts_uj': total_ops_uj,
        'E_Addr_uj': total_addr_uj,
        'E_Mem_uj': total_mem_uj,
        'E_Total_uj': grand_total,
        'layer_details': layer_energies,
    }


# ============================================================================
# 4. SpikF ENERGY (for comparison)
# ============================================================================
def compute_spikf_energy(firing_rate, T, data, pred_len):
    """Compute SpikF energy using S-FFT/S-iFFT functions from calculate.py."""
    D = DATA_DIM[data]
    N = 32  # SpikF default patch_num
    p_d = 32  # SpikF default patch_dim
    h_dim = 720
    H = pred_len
    
    # S-FFT energy
    e_sfft = sfft_sops(L=N, alpha=firing_rate, T=T, dim_1=p_d, dim_2=D)
    
    # S-iFFT energy
    e_sifft = sifft_sops(L=N, alpha=firing_rate, T=T, dim_1=p_d, dim_2=D)
    
    # Selector: Linear(N -> N)
    sel_spikes = int(firing_rate * T * N * p_d * D)
    e_sel = fnn_snn(T=T, N_out=N, position_1=p_d, position_2=D,
                    input_spikes=sel_spikes, output_spikes=sel_spikes,
                    bias=True, lif_leak=True)
    
    # Intra-conv and inter-conv (Conv2d)
    # Simplified as linear operations
    e_intra = fnn_snn(T=T, N_out=p_d, position_1=N, position_2=D,
                      input_spikes=sel_spikes, output_spikes=sel_spikes,
                      bias=True, lif_leak=True)
    e_inter = fnn_snn(T=T, N_out=p_d, position_1=N, position_2=D,
                      input_spikes=sel_spikes, output_spikes=sel_spikes,
                      bias=True, lif_leak=True)
    
    # SPE
    spe_in = 96 // N
    spe_spikes = int(firing_rate * T * spe_in * D)
    e_spe = fnn_snn(T=T, N_out=p_d, position_1=N, position_2=D,
                    input_spikes=spe_spikes, output_spikes=sel_spikes,
                    bias=True, lif_leak=True)
    
    # Decoder (ANN)
    dec_flops = N * p_d * h_dim + h_dim * H
    e_dec = dec_flops * PJ_PER_MAC * 1e-6
    
    total = e_sfft + e_sifft + e_sel + e_intra + e_inter + e_spe + e_dec
    
    return {
        'E_SFFT_uj': e_sfft,
        'E_SiFFT_uj': e_sifft,
        'E_Selector_uj': e_sel,
        'E_IntraConv_uj': e_intra,
        'E_InterConv_uj': e_inter,
        'E_SPE_uj': e_spe,
        'E_Decoder_uj': e_dec,
        'E_Total_uj': total,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh2',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange', 'weather'])
    parser.add_argument('--pred_len', type=int, default=96)
    args = parser.parse_args()
    
    os.makedirs('energy_results', exist_ok=True)
    
    T = 16  # spiking timesteps
    
    results = {}
    
    # ---- SwaveNet ----
    print(f"\n{'='*60}")
    print(f"SwaveNet Energy Calculation — {args.data} pl={args.pred_len}")
    print(f"{'='*60}")
    
    # sw_args = create_args(args.data, args.pred_len, 'abl_noDWC', 0)
    # exp = Exp_ETT(sw_args)
    # test_loader = exp._get_data(flag='test')

    sw_args = create_args(args.data, args.pred_len, 'abl_noDWC', 0)
    exp = Exp_ETT(sw_args)
    ckpt = f"test_results/{args.data}/noDWC_final/abl_noDWC_{args.data}_ftM_sl96_pl{args.pred_len}_lr0.0005_bs32_run0/checkpoint.pth"
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"Loaded SwaveNet checkpoint: {ckpt}")
    test_loader = exp._get_data(flag='test')
    
    # Measure firing rates
    print("Measuring firing rates...")
    fr_per_layer, avg_fr = measure_firing_rates(exp.model, test_loader, exp.device)
    print(f"Average firing rate: {avg_fr:.4f}")
    
    # Simple energy
    simple = compute_simple_energy(exp.model, avg_fr, T, args.data, args.pred_len)
    print(f"\n--- Simple SOPs Energy ---")
    print(f"  FLOPs:     {simple['total_flops']:,.0f}")
    print(f"  SOPs:      {simple['total_sops']:,.0f}")
    print(f"  E_ANN:     {simple['E_ann_uj']:.4f} µJ")
    print(f"  E_SNN:     {simple['E_snn_uj']:.4f} µJ")
    print(f"  Ratio:     {simple['ratio_ann_over_snn']:.2f}x")
    
    # Detailed energy
    detailed = compute_detailed_energy(exp.model, fr_per_layer, T, args.data, args.pred_len)
    print(f"\n--- Detailed Energy (Lemaire et al.) ---")
    print(f"  E_Opts:    {detailed['E_Opts_uj']:.4f} µJ")
    print(f"  E_Addr:    {detailed['E_Addr_uj']:.4f} µJ")
    print(f"  E_Mem:     {detailed['E_Mem_uj']:.4f} µJ")
    print(f"  E_Total:   {detailed['E_Total_uj']:.4f} µJ")
    
    results['SwaveNet'] = {
        'firing_rate': avg_fr,
        'firing_rates_per_layer': {k: v['avg_rate'] for k, v in fr_per_layer.items()},
        'simple': {k: v for k, v in simple.items() if k != 'layer_details'},
        'detailed': {k: v for k, v in detailed.items() if k != 'layer_details'},
    }
    
    del exp
    torch.cuda.empty_cache()
    
    # ---- SpikF ----
    print(f"\n{'='*60}")
    print(f"SpikF Energy Calculation — {args.data} pl={args.pred_len}")
    print(f"{'='*60}")
    
    # sp_args = create_args(args.data, args.pred_len, 'SpikF', 0)
    # exp_sp = Exp_ETT(sp_args)
    # test_loader_sp = exp_sp._get_data(flag='test')

    sp_args = create_args(args.data, args.pred_len, 'SpikF', 0)
    exp_sp = Exp_ETT(sp_args)
    ckpt_sp = f"test_results/{args.data}/SpikF_baseline/SpikF_{args.data}_ftM_sl96_pl{args.pred_len}_lr0.0005_bs32_run0/checkpoint.pth"
    #ckpt_sp = f"test_results/{args.data}/SpikF_baseline/SpikF_baseline_{args.data}_pl{args.pred_len}/checkpoint.pth"
    if os.path.exists(ckpt_sp):
        exp_sp.model.load_state_dict(torch.load(ckpt_sp, map_location=exp_sp.device))
        print(f"Loaded SpikF checkpoint: {ckpt_sp}")
    else:
        print(f"WARNING: SpikF checkpoint not found: {ckpt_sp}")
    test_loader_sp = exp_sp._get_data(flag='test')

    
    fr_per_layer_sp, avg_fr_sp = measure_firing_rates(exp_sp.model, test_loader_sp, exp_sp.device)
    print(f"Average firing rate: {avg_fr_sp:.4f}")
    
    simple_sp = compute_simple_energy(exp_sp.model, avg_fr_sp, T, args.data, args.pred_len)
    print(f"\n--- Simple SOPs Energy ---")
    print(f"  FLOPs:     {simple_sp['total_flops']:,.0f}")
    print(f"  SOPs:      {simple_sp['total_sops']:,.0f}")
    print(f"  E_ANN:     {simple_sp['E_ann_uj']:.4f} µJ")
    print(f"  E_SNN:     {simple_sp['E_snn_uj']:.4f} µJ")
    print(f"  Ratio:     {simple_sp['ratio_ann_over_snn']:.2f}x")
    
    spikf_detailed = compute_spikf_energy(avg_fr_sp, T, args.data, args.pred_len)
    print(f"\n--- SpikF Component Energy ---")
    for k, v in spikf_detailed.items():
        if k != 'E_Total_uj':
            print(f"  {k}: {v:.4f} µJ")
    print(f"  E_Total:   {spikf_detailed['E_Total_uj']:.4f} µJ")
    
    results['SpikF'] = {
        'firing_rate': avg_fr_sp,
        'simple': {k: v for k, v in simple_sp.items() if k != 'layer_details'},
        'spikf_detailed': spikf_detailed,
    }
    
    del exp_sp
    torch.cuda.empty_cache()
    
    # ---- Comparison ----
    print(f"\n{'='*60}")
    print(f"ENERGY COMPARISON — {args.data} pl={args.pred_len}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'SwaveNet':>15} {'SpikF':>15}")
    print("-" * 55)
    print(f"{'Firing Rate':<25} {avg_fr:>15.4f} {avg_fr_sp:>15.4f}")
    print(f"{'FLOPs':<25} {simple['total_flops']:>15,.0f} {simple_sp['total_flops']:>15,.0f}")
    print(f"{'SOPs':<25} {simple['total_sops']:>15,.0f} {simple_sp['total_sops']:>15,.0f}")
    print(f"{'E_SNN (simple) µJ':<25} {simple['E_snn_uj']:>15.4f} {simple_sp['E_snn_uj']:>15.4f}")
    print(f"{'ANN/SNN ratio':<25} {simple['ratio_ann_over_snn']:>15.2f}x {simple_sp['ratio_ann_over_snn']:>15.2f}x")
    
    # Save
    outfile = f"energy_results/{args.data}_pl{args.pred_len}_energy.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == '__main__':
    main()
