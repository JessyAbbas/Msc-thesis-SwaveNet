"""
SwaveNet Visualizations (v2 - with checkpoint loading)
Usage: CUDA_VISIBLE_DEVICES=0 python visualizations_v2.py --data ETTh2 --pred_len 96
"""
import os, sys, argparse, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':12,'font.family':'serif','axes.labelsize':14,'axes.titlesize':14,'legend.fontsize':11,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp.exp_ETT_new import Exp_ETT
from spikingjelly.clock_driven import functional

BEST_PARAMS = {
    'ETTh1':{'hidden_dim':540,'patch_dim':32,'levels':3,'patch_num':32},
    'ETTh2':{'hidden_dim':720,'patch_dim':32,'levels':2,'patch_num':16},
    'ETTm1':{'hidden_dim':180,'patch_dim':64,'levels':2,'patch_num':32},
    'ETTm2':{'hidden_dim':180,'patch_dim':64,'levels':3,'patch_num':8},
    'exchange':{'hidden_dim':180,'patch_dim':64,'levels':1,'patch_num':32},
    'weather':{'hidden_dim':720,'patch_dim':64,'levels':3,'patch_num':8},
}
DATA_PATH = {'ETTh1':'ETTh1.csv','ETTh2':'ETTh2.csv','ETTm1':'ETTm1.csv','ETTm2':'ETTm2.csv','exchange':'exchange_rate.csv','weather':'weather.csv'}

def get_swavenet_ckpt(data, pred_len):
    return f"test_results/{data}/noDWC_final/abl_noDWC_{data}_ftM_sl96_pl{pred_len}_lr0.0005_bs32_run0/checkpoint.pth"
def get_spikf_ckpt(data, pred_len):
    return f"test_results/{data}/SpikF_baseline/SpikF_{data}_ftM_sl96_pl{pred_len}_lr0.0005_bs32_run0/checkpoint.pth"

def create_args(data, pred_len, model, gpu=0):
    params = BEST_PARAMS[data] if model != 'SpikF' else {'hidden_dim':720,'patch_dim':32,'levels':1,'patch_num':32}
    return argparse.Namespace(data=data,root_path='/media/homes/abbasj/SpikF/datasets/long/',data_path=DATA_PATH[data],features='M',target='OT',cols=None,seq_len=96,label_len=0,pred_len=pred_len,hidden_dim=params['hidden_dim'],patch_dim=params['patch_dim'],levels=params['levels'],patch_num=params['patch_num'],T=16,tau=2.0,alpha=0.5,batch_size=32,lr=5e-4,train_epochs=0,patience=5,loss='mae',model=model,keep_levels='all',num_heads=8,use_gray_pe=1,use_log_pe=1,gray_num_bits=10,dwc_kernel=5,num_workers=4,rank=gpu,save=False,model_name='viz',checkpoints='viz_ckpt',mean=0,last=0,std=0,scale_select='both',wavelet='haar',J=1)

def load_ckpt(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  Loaded: {path}")
        return True
    print(f"  WARNING: Not found: {path}")
    return False

def run_inference(exp, test_loader, pred_len):
    exp.model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, (list, tuple)):
                bx = batch_data[0].float().to(exp.device)
                by = batch_data[1].float().to(exp.device)
            else:
                bx = batch_data.float().to(exp.device); by = None
            functional.reset_net(exp.model)
            out = exp.model(bx)
            if out.dim() == 4: out = out.mean(dim=0)
            all_preds.append(out.detach().cpu().numpy())
            if by is not None: all_trues.append(by.detach().cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0) if all_trues else None
    return preds, trues

def plot_predictions(data, pred_len, model_name, gpu, num_samples=3, variables=[0]):
    args = create_args(data, pred_len, model_name, gpu)
    exp = Exp_ETT(args)
    if not load_ckpt(exp.model, get_swavenet_ckpt(data, pred_len), exp.device):
        del exp; torch.cuda.empty_cache(); return
    test_loader = exp._get_data(flag='test')
    preds, trues = run_inference(exp, test_loader, pred_len)
    print(f"  Preds: {preds.shape}, Trues: {trues.shape if trues is not None else 'None'}")
    outdir = f'visualizations/{data}'; os.makedirs(outdir, exist_ok=True)
    if trues is None: del exp; torch.cuda.empty_cache(); return
    np.random.seed(42); n = min(len(preds), len(trues))
    idx = sorted(np.random.choice(n, min(num_samples, n), replace=False))
    for vi in variables:
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples), sharex=True)
        if num_samples == 1: axes = [axes]
        for i, (ax, si) in enumerate(zip(axes, idx)):
            t = np.arange(pred_len)
            tv = trues[si, :pred_len, vi] if trues.ndim==3 else trues[si, :pred_len]
            pv = preds[si, :pred_len, vi] if preds.ndim==3 else preds[si, :pred_len]
            ax.plot(t, tv, 'b-', lw=1.5, label='Ground Truth', alpha=0.8)
            ax.plot(t, pv, 'r--', lw=1.5, label='SwaveNet', alpha=0.8)
            ax.set_ylabel(f'Sample {si}'); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Prediction Horizon (steps)')
        fig.suptitle(f'{data} - Prediction vs Ground Truth (Var {vi}, H={pred_len})', y=1.02)
        plt.tight_layout()
        fname = f'{outdir}/pred_vs_true_{data}_pl{pred_len}_var{vi}.png'
        plt.savefig(fname); plt.close(); print(f"  Saved: {fname}")
    fig, ax = plt.subplots(figsize=(14, 4))
    for i in range(min(5, n)):
        to = i * pred_len; t = np.arange(to, to + pred_len)
        tv = trues[i, :pred_len, variables[0]] if trues.ndim==3 else trues[i, :pred_len]
        pv = preds[i, :pred_len, variables[0]] if preds.ndim==3 else preds[i, :pred_len]
        ax.plot(t, tv, 'b-', lw=1.2, alpha=0.7, label='Ground Truth' if i==0 else None)
        ax.plot(t, pv, 'r--', lw=1.2, alpha=0.7, label='SwaveNet' if i==0 else None)
        if i > 0: ax.axvline(x=to, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Time Steps'); ax.set_ylabel('Value')
    ax.set_title(f'{data} - Consecutive Predictions (Var {variables[0]}, H={pred_len})')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3); plt.tight_layout()
    fname = f'{outdir}/continuous_pred_{data}_pl{pred_len}.png'
    plt.savefig(fname); plt.close(); print(f"  Saved: {fname}")
    del exp; torch.cuda.empty_cache()

def analyze_firing_rates(data, pred_len, model_name, gpu, num_batches=10):
    args = create_args(data, pred_len, model_name, gpu)
    exp = Exp_ETT(args)
    if not load_ckpt(exp.model, get_swavenet_ckpt(data, pred_len), exp.device):
        del exp; torch.cuda.empty_cache(); return None
    test_loader = exp._get_data(flag='test')
    firing_rates = {}; hooks = []
    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                r = out.abs().float().mean().item()
                if name not in firing_rates: firing_rates[name] = []
                firing_rates[name].append(r)
        return hook_fn
    from spikingjelly.clock_driven.neuron import MultiStepLIFNode
    for name, module in exp.model.named_modules():
        if isinstance(module, MultiStepLIFNode):
            hooks.append(module.register_forward_hook(make_hook(name)))
        if hasattr(module, 'neuronal_fire') and 'Neg' in type(module).__name__:
            hooks.append(module.register_forward_hook(make_hook(f'{name}(ternary)')))
    exp.model.eval(); bc = 0
    with torch.no_grad():
        for bd in test_loader:
            if bc >= num_batches: break
            bx = bd[0].float().to(exp.device) if isinstance(bd, (list,tuple)) else bd.float().to(exp.device)
            functional.reset_net(exp.model); _ = exp.model(bx); bc += 1
    for h in hooks: h.remove()
    outdir = f'visualizations/{data}'; os.makedirs(outdir, exist_ok=True)
    lnames, mrates, srates = [], [], []
    print(f"\n{'='*60}\nFiring Rates - {model_name} | {data} | pl={pred_len}\n{'='*60}")
    print(f"{'Layer':<50} {'Rate':>8} {'Std':>8}"); print("-"*70)
    for name, rates in sorted(firing_rates.items()):
        m, s = np.mean(rates), np.std(rates)
        sn = name.replace('blocks.','B').replace('.sws_block.','.sws.').replace('.light_spikformer.','.lsb.')
        lnames.append(sn if len(sn)<40 else sn[-40:]); mrates.append(m); srates.append(s)
        print(f"{name:<50} {m:>8.4f} {s:>8.4f}")
    if not mrates: print("  No spiking layers!"); del exp; torch.cuda.empty_cache(); return None
    avg = np.mean(mrates); print(f"\nAverage: {avg:.4f}")
    fig, ax = plt.subplots(figsize=(max(10, len(lnames)*0.6), 6))
    x = np.arange(len(lnames))
    ax.bar(x, mrates, yerr=srates, capsize=3, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.axhline(y=avg, color='red', ls='--', alpha=0.7, label=f'Average: {avg:.3f}')
    ax.set_xticks(x); ax.set_xticklabels(lnames, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Firing Rate'); ax.set_title(f'Layer-wise Firing Rates - {data} (H={pred_len})')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    fname = f'{outdir}/firing_rates_{data}_pl{pred_len}.png'
    plt.savefig(fname); plt.close(); print(f"  Saved: {fname}")
    res = {'model':model_name,'data':data,'pred_len':pred_len,'overall_avg_firing_rate':float(avg),
           'layers':{n:{'mean':float(np.mean(r)),'std':float(np.std(r))} for n,r in firing_rates.items()}}
    with open(f'{outdir}/firing_rates_{data}_pl{pred_len}.json','w') as f: json.dump(res,f,indent=2)
    del exp; torch.cuda.empty_cache(); return res

def plot_comparison(data, pred_len, gpu, variable=0):
    outdir = f'visualizations/{data}'; os.makedirs(outdir, exist_ok=True)
    results = {}
    for mn in ['abl_noDWC', 'SpikF']:
        args = create_args(data, pred_len, mn, gpu)
        exp = Exp_ETT(args)
        ckpt = get_swavenet_ckpt(data, pred_len) if mn == 'abl_noDWC' else get_spikf_ckpt(data, pred_len)
        load_ckpt(exp.model, ckpt, exp.device)
        test_loader = exp._get_data(flag='test')
        preds, trues = run_inference(exp, test_loader, pred_len)
        results[mn] = {'preds': preds, 'trues': trues}
        del exp; torch.cuda.empty_cache()
    trues = results['abl_noDWC']['trues']
    ps = results['abl_noDWC']['preds']; pf = results['SpikF']['preds']
    if trues is None: print("  No ground truth."); return
    np.random.seed(42); n = min(len(trues), len(ps), len(pf))
    idx = sorted(np.random.choice(n, min(3, n), replace=False))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, (ax, si) in enumerate(zip(axes, idx)):
        t = np.arange(pred_len)
        if trues.ndim==3:
            tv=trues[si,:pred_len,variable]; sv=ps[si,:pred_len,variable]; fv=pf[si,:pred_len,variable]
        else:
            tv=trues[si,:pred_len]; sv=ps[si,:pred_len]; fv=pf[si,:pred_len]
        ax.plot(t, tv, 'b-', lw=1.5, label='Ground Truth', alpha=0.8)
        ax.plot(t, sv, 'r--', lw=1.5, label='SwaveNet', alpha=0.8)
        ax.plot(t, fv, 'g:', lw=1.5, label='SpikF', alpha=0.7)
        ax.set_ylabel(f'Sample {si}'); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Prediction Horizon (steps)')
    fig.suptitle(f'{data} - SwaveNet vs SpikF (Var {variable}, H={pred_len})', y=1.02)
    plt.tight_layout()
    fname = f'{outdir}/comparison_{data}_pl{pred_len}.png'
    plt.savefig(fname); plt.close(); print(f"  Saved: {fname}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh2', choices=['ETTh1','ETTh2','ETTm1','ETTm2','exchange','weather'])
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--mode', type=str, default='all', choices=['all','pred','firing','compare'])
    args = parser.parse_args()
    if args.mode in ['all','pred']:
        print("\n=== Prediction Plots ===")
        plot_predictions(args.data, args.pred_len, 'abl_noDWC', 0, num_samples=3, variables=[0])
    if args.mode in ['all','firing']:
        print("\n=== Firing Rate Analysis ===")
        analyze_firing_rates(args.data, args.pred_len, 'abl_noDWC', 0)
    if args.mode in ['all','compare']:
        print("\n=== SwaveNet vs SpikF Comparison ===")
        plot_comparison(args.data, args.pred_len, 0)
    print("\nDone! Check visualizations/ folder.")

if __name__ == '__main__':
    main()
