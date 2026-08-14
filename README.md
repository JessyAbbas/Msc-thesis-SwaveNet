# SwaveNet: Spiking Wavelet Network for Energy-Efficient Time-Series Forecasting

SwaveNet replaces FFT-based frequency selection in spiking neural networks with Haar Discrete Wavelet Transform using ternary spiking neurons {−1, 0, +1}, combined with XNOR-based spike-driven self-attention and a novel combination of Gray Code and Logarithmic positional encoding.

## Architecture

<p align="center">
  <img src="figures/architecture.png" width="800"/>
</p>

```
Input → SPE → [DB-SWB → LSB] × ℓ → MLP Decoder → Output
```

- **DB-SWB** (Dual-Branch Spiking Wavelet Block): Spiking Wavelet Selector (SWS) + Wavelet-Guided Modulation (WGM) → LIF-enrich fusion
- **LSB** (Light Spikformer Block): XNOR-SSA with combined Gray-PE + Log-PE

## Key Features

- Haar DWT with ternary spikes replaces FFT for spike-compatible frequency decomposition
- Dual-branch wavelet processing: fine-grained selection (SWS) + coarse modulation (WGM)
- Combined Gray-PE + Log-PE for spike-driven attention (proposed separately in [SeqSNN](https://github.com/microsoft/SeqSNN), combined here for the first time)
- 2.9× more energy efficient than SpikF on neuromorphic hardware estimates
- 45% fewer parameters, 38% lower firing rates than SpikF

## Installation

```bash
git clone https://github.com/[username]/SwaveNet.git
cd SwaveNet
pip install torch spikingjelly==0.0.0.0.14 optuna numpy pandas matplotlib
```

Requires Python ≥ 3.11, PyTorch ≥ 2.0, CUDA-compatible GPU (≥ 8GB).

## Dataset Setup

Download datasets into `datasets/long/`: [ETT](https://github.com/zhouhaoyi/ETDataset), [Exchange](https://github.com/laiguokun/multivariate-time-series-data), [Weather](https://www.bgc-jena.mpg.de/wetter/).

## Usage

```bash
# Train SwaveNet
CUDA_VISIBLE_DEVICES=0 python run_long.py \
    --data ETTh2 --root_path datasets/long/ --data_path ETTh2.csv \
    --features M --seq_len 96 --pred_len 96 --batch_size 32 --lr 5e-4 \
    --levels 2 --patch_dim 32 --patch_num 16 --T 16 --hidden_dim 720 \
    --train_epochs 20 --patience 5 --model abl_noDWC \
    --keep_levels all --num_heads 8 --use_gray_pe 1 --use_log_pe 1 \
    --gray_num_bits 10 --dwc_kernel 5 --save --model_name SwaveNet_ETTh2

# Train SpikF baseline
CUDA_VISIBLE_DEVICES=0 python run_long.py \
    --data ETTh2 --root_path datasets/long/ --data_path ETTh2.csv \
    --features M --seq_len 96 --pred_len 96 --batch_size 32 --lr 5e-4 \
    --levels 1 --patch_dim 32 --patch_num 32 --T 16 --hidden_dim 720 \
    --train_epochs 20 --patience 5 --model SpikF --save --model_name SpikF_ETTh2

# Hyperparameter tuning
CUDA_VISIBLE_DEVICES=0 python optuna_tuning.py --data ETTh2 --gpu 0 --n_trials 72

# Energy calculation
CUDA_VISIBLE_DEVICES=0 python energy_calculation.py --data ETTh2 --pred_len 96

# Visualizations
CUDA_VISIBLE_DEVICES=0 python visualizations_v2.py --data ETTh2 --pred_len 96
```

## Project Structure

```
SwaveNet/
├── models/
│   ├── abl_noDWC.py              # SwaveNet (final model)
│   ├── SpikF.py                  # SpikF baseline
│   └── abl_*.py                  # Ablation variants
├── exp/
│   └── exp_ETT_new.py            # Experiment runner
├── data_provider/
│   └── ETT_data_loader.py
├── run_long.py                   # Training
├── optuna_tuning.py              # Hyperparameter tuning
├── energy_calculation.py         # Energy analysis
├── visualizations_v2.py          # Plots and firing rates
├── timing_comparison.py          # Inference timing
└── calculate.py                  # Energy utilities (Lemaire et al.)
```

## Acknowledgements

Built upon [SpikF](https://github.com/[spikf-repo]) [Wu et al., 2025], [SWformer](https://github.com/[swformer-repo]) [Fang et al., 2024], [SeqSNN](https://github.com/microsoft/SeqSNN) [Lv et al., 2025], and [SpikingJelly](https://github.com/fangwei123456/spikingjelly).

## Citation

```bibtex
@mastersthesis{abbas2026swavenet,
    title={SwaveNet: Spiking Wavelet Network for Energy-Efficient Time-Series Forecasting},
    author={Abbas, J.},
    year={2026},
    school={[University Name]}
}
```
