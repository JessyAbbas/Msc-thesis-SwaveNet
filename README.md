# SwaveNet: Spiking Wavelet Network for Energy-Efficient Time-Series Forecasting

> **SwaveNet** replaces FFT-based frequency selection in spiking neural networks with Haar Discrete Wavelet Transform using ternary spiking neurons, combined with XNOR-based spike-driven self-attention and a novel combination of Gray Code and Logarithmic positional encoding.

## Architecture

```
Input → SPE → [DB-SWB → LSB] × ℓ → MLP Decoder → Output
```

| Component | Description |
|-----------|-------------|
| **SPE** | Spiking Patch Encoder — converts continuous time-series to binary spike trains |
| **DB-SWB** | Dual-Branch Spiking Wavelet Block |
| ├─ **SWS** | Spiking Wavelet Selector — data-dependent wavelet coefficient selection via learned binary mask |
| └─ **WGM** | Wavelet-Guided Modulation — coarse spectral energy summary via linear projection |
| **LSB** | Light Spikformer Block — XNOR-SSA attention with combined Gray-PE + Log-PE |
| **MLP Decoder** | Two-layer MLP with instance normalisation |

<p align="center">
  <img src="figures/architecture.png" width="800"/>
</p>

## Key Features

- **Haar DWT with ternary spikes {−1, 0, +1}** — spike-compatible multi-resolution frequency decomposition
- **Dual-branch wavelet processing** — fine-grained selection (SWS) + coarse modulation (WGM)
- **Combined Gray-PE + Log-PE** — novel combination of positional encodings for spike-driven attention (proposed independently in [SeqSNN](https://github.com/microsoft/SeqSNN), combined here for the first time)
- **2.9× more energy efficient** than SpikF baseline on neuromorphic hardware estimates
- **38% lower firing rates** than SpikF through progressive spike sparsification

## Results

### Long-term Forecasting (MSE, lower is better)

| Dataset | Horizon | SwaveNet | SpikF | Improvement |
|---------|---------|----------|-------|-------------|
| ETTh1 | 96 | **0.3743** | 0.3806 | -1.7% |
| ETTh1 | 336 | **0.4697** | 0.4805 | -2.2% |
| ETTh2 | 96 | **0.2832** | 0.2910 | -2.7% |
| ETTh2 | 336 | **0.4094** | 0.4171 | -1.8% |
| ETTm1 | 96 | **0.3084** | 0.3158 | -2.3% |
| ETTm1 | 336 | **0.3854** | 0.3991 | -3.4% |
| ETTm2 | 96 | **0.1708** | 0.1749 | -2.3% |
| ETTm2 | 336 | **0.2949** | 0.3005 | -1.9% |
| Exchange | 96 | **0.0857** | 0.0866 | -1.0% |
| Weather | 96 | **0.1632** | 0.1634 | -0.1% |

### Energy Efficiency (ETTh2, H=96, 45nm CMOS)

| Metric | SwaveNet | SpikF |
|--------|----------|-------|
| Parameters | 448K | 816K |
| Avg Firing Rate | 0.136 | 0.220 |
| E_SNN (µJ) | **0.874** | 2.554 |
| Energy Ratio | **2.9× more efficient** | — |

## Installation

```bash
git clone https://github.com/[username]/SwaveNet.git
cd SwaveNet

# Create conda environment
conda create -n swavenet python=3.11
conda activate swavenet

# Install dependencies
pip install torch torchvision torchaudio
pip install spikingjelly==0.0.0.0.14
pip install optuna
pip install numpy pandas matplotlib

# Verify installation
python -c "import spikingjelly; print('SpikingJelly OK')"
```

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- SpikingJelly 0.0.0.0.14 (clock_driven API)
- CUDA-compatible GPU (≥ 8GB VRAM for ETT datasets, ≥ 16GB for large datasets)
- Optuna (for hyperparameter tuning)

## Dataset Setup

Download datasets and place in `datasets/long/`:

```
datasets/long/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── exchange_rate.csv
└── weather.csv
```

ETT datasets: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

| Dataset | Variables | Frequency | Train/Val/Test |
|---------|----------|-----------|----------------|
| ETTh1/h2 | 7 | 1 Hour | 8545/2881/2881 |
| ETTm1/m2 | 7 | 15 Min | 34465/11521/11521 |
| Exchange | 8 | 1 Day | 5120/665/1422 |
| Weather | 21 | 10 Min | 36792/5271/10540 |

## Usage

### Training

```bash
# SwaveNet on ETTh2 (best params from Optuna)
CUDA_VISIBLE_DEVICES=0 python run_long.py \
    --data ETTh2 \
    --root_path datasets/long/ \
    --data_path ETTh2.csv \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 32 \
    --lr 5e-4 \
    --levels 2 \
    --patch_dim 32 \
    --patch_num 16 \
    --T 16 \
    --hidden_dim 720 \
    --train_epochs 20 \
    --patience 5 \
    --model abl_noDWC \
    --keep_levels all \
    --num_heads 8 \
    --use_gray_pe 1 \
    --use_log_pe 1 \
    --gray_num_bits 10 \
    --dwc_kernel 5 \
    --save \
    --model_name SwaveNet_ETTh2_pl96
```

### SpikF Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python run_long.py \
    --data ETTh2 \
    --root_path datasets/long/ \
    --data_path ETTh2.csv \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --batch_size 32 \
    --lr 5e-4 \
    --levels 1 \
    --patch_dim 32 \
    --patch_num 32 \
    --T 16 \
    --hidden_dim 720 \
    --train_epochs 20 \
    --patience 5 \
    --model SpikF \
    --save \
    --model_name SpikF_ETTh2_pl96
```

### Hyperparameter Tuning

```bash
# Optuna TPE tuning (72 trials, single seed, pred_len=336)
CUDA_VISIBLE_DEVICES=0 python optuna_tuning.py \
    --data ETTh2 \
    --gpu 0 \
    --n_trials 72
```

### Energy Calculation

```bash
# Theoretical energy consumption (requires trained checkpoint)
CUDA_VISIBLE_DEVICES=0 python energy_calculation.py \
    --data ETTh2 \
    --pred_len 96
```

### Visualizations

```bash
# Prediction plots, firing rates, SwaveNet vs SpikF comparison
CUDA_VISIBLE_DEVICES=0 python visualizations_v2.py \
    --data ETTh2 \
    --pred_len 96

# Specific mode
CUDA_VISIBLE_DEVICES=0 python visualizations_v2.py \
    --data ETTh2 \
    --pred_len 96 \
    --mode firing  # Options: all, pred, firing, compare
```

### Inference Timing

```bash
CUDA_VISIBLE_DEVICES=0 python timing_comparison.py \
    --data ETTh2 \
    --pred_lens 96 336
```

## Best Hyperparameters

Found via Optuna TPE sampling on pred_len=336:

| Dataset | hidden_dim | patch_dim | levels (ℓ) | patch_num | Tuning MSE |
|---------|-----------|-----------|------------|-----------|------------|
| ETTh1 | 540 | 32 | 3 | 32 | 0.4669 |
| ETTh2 | 720 | 32 | 2 | 16 | 0.4035 |
| ETTm1 | 180 | 64 | 2 | 32 | 0.3842 |
| ETTm2 | 180 | 64 | 3 | 8 | 0.2940 |
| Exchange | 180 | 64 | 1 | 32 | 0.3488 |
| Weather | 720 | 64 | 3 | 8 | 0.2652 |

## Project Structure

```
SwaveNet/
├── models/
│   ├── abl_noDWC.py              # Final SwaveNet model (no DWC)
│   ├── SpikF.py                  # SpikF baseline
│   ├── SwaveNetV1_LightSpikformer.py  # SwaveNet with DWC (development)
│   ├── abl_noWavelet.py          # Ablation: remove both wavelet branches
│   ├── abl_noBranch1.py          # Ablation: remove SWS selector
│   ├── abl_noBranch2.py          # Ablation: remove WGM modulation
│   ├── abl_noSpike.py            # Ablation: float DWT (no ternary)
│   ├── abl_noAttention.py        # Ablation: replace attention with conv
│   └── abl_vth05.py              # Ablation: enrich_lif threshold 0.5
├── exp/
│   └── exp_ETT_new.py            # Experiment runner
├── data_provider/
│   └── ETT_data_loader.py        # Dataset loading
├── datasets/long/                # Dataset CSV files
├── scripts/                      # Training shell scripts
├── run_long.py                   # Main training script
├── optuna_tuning.py              # Hyperparameter tuning
├── energy_calculation.py         # Energy efficiency analysis
├── timing_comparison.py          # Inference timing
├── visualizations_v2.py          # Prediction & firing rate plots
├── calculate.py                  # Energy calculation utilities
└── figures/
    └── architecture.png          # Architecture diagram
```

## Training Protocol

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Early Stopping | patience=5, min_delta=0.0002 |
| Optimiser | Adam (β₁=0.9, β₂=0.999) |
| Learning Rate | 5×10⁻⁴ |
| LR Scheduler | CosineAnnealingLR (T_max=20) |
| Loss | MAE (L1) |
| Spiking Timesteps (T_s) | 16 |
| Membrane Time Constant (τ) | 2.0 |
| Surrogate Gradient | Sigmoid (α=4.0) |
| Seeds | [2022, 2023, 2024] |
| Ternary Threshold (V_th) | 2.0 |

## Ablation Study

Conducted on ETTh2 and ETTm1 at horizons 96 and 336:

| Configuration | ETTh2-96 | ETTh2-336 | ETTm1-96 | ETTm1-336 |
|--------------|----------|-----------|----------|-----------|
| Full Model | 0.2834 | 0.4077 | 0.3063 | 0.3862 |
| w/o Wavelet | 0.2840 | 0.4089 | 0.3101 | 0.3864 |
| w/o Branch 1 (SWS) | 0.2836 | 0.4064 | 0.3065 | 0.3833 |
| w/o Branch 2 (WGM) | 0.2838 | 0.4079 | 0.3062 | 0.3852 |
| w/o Spike (float) | 0.2839 | 0.4072 | 0.3118 | 0.3853 |
| w/o Attention | 0.2826 | 0.4115 | 0.3055 | 0.3901 |
| V_th=0.5 | 0.2824 | 0.4081 | 0.3081 | 0.3832 |

Key findings: Ternary spiking provides +1.8% regularisation benefit on ETTm1. Attention is essential for long horizons (+0.9-1.0% at pl=336). Dual branches are complementary (removing both hurts more than removing either alone).

## Acknowledgements

This work builds upon:
- [SpikF](https://github.com/[spikf-repo]) — Spiking Fourier Network [Wu et al., 2025]
- [SWformer](https://github.com/[swformer-repo]) — Spiking Wavelet Transformer [Fang et al., 2024]
- [SeqSNN](https://github.com/microsoft/SeqSNN) — Relative Positional Encoding for Spiking Transformers [Lv et al., 2025]
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) — SNN framework [Fang et al., 2023]

## Citation

```bibtex
@mastersthesis{abbas2026swavenet,
    title={SwaveNet: Spiking Wavelet Network for Energy-Efficient Time-Series Forecasting},
    author={Abbas, J.},
    year={2026},
    school={[University Name]}
}
```

## License

This project is for academic research purposes.
