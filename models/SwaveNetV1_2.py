# SwaveNetV1_2: SpikF-style Spiking Wavelet Selector with db/sym wavelets
# Uses pytorch_wavelets library instead of Haar matrix
# Supports: db4, db8, sym4, sym8

import math
from abc import abstractmethod
from typing import Callable, Iterable, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

from spikingjelly.clock_driven import surrogate, base
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from pytorch_wavelets import DWT1DForward, DWT1DInverse


# ============================================================================
# TERNARY SPIKING NEURONS (from SWformer)
# ============================================================================

class NegBaseNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        neg_v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
    ):
        super().__init__()
        assert isinstance(v_threshold, float)
        assert isinstance(neg_v_threshold, float)
        assert (isinstance(v_reset, float) or v_reset is None)

        if v_reset is None:
            self.register_memory("v", 0.0)
        else:
            self.register_memory("v", v_reset)

        self.register_memory("v_threshold", v_threshold)
        self.register_memory("neg_v_threshold", neg_v_threshold)
        self.register_memory("v_reset", v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.register_memory("inhibit", 1e-3)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        pos_spike = self.surrogate_function(self.v - self.v_threshold)
        neg_spike = self.surrogate_function(-(self.v + self.neg_v_threshold + self.inhibit))
        return pos_spike - neg_spike  # ∈ {-1, 0, +1}

    def neuronal_reset(self, spike: torch.Tensor):
        spike_d = spike.detach() if self.detach_reset else spike
        pos = spike_d > 0
        neg = spike_d < 0
        spike_d_vth = pos * self.v_threshold + neg * self.neg_v_threshold

        if self.v_reset is None:
            self.v = self.v - spike_d_vth
        else:
            self.v = (1.0 - spike_d.abs()) * self.v + spike_d * self.v_reset

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        s = self.neuronal_fire()
        self.neuronal_reset(s)
        return s


class NegIFNode(NegBaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        neg_v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
    ):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class MultiStepNegIFNode(NegIFNode):
    """Multi-step ternary neuron: input shape [T, ...]."""

    def __init__(
        self,
        v_threshold: float = 1.0,
        neg_v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        backend: str = "torch",
    ):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory("v_seq", None)
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1, "MultiStepNegIFNode expects x_seq with shape [T, ...]"
        spike_seq = []
        v_seq = []
        for t in range(x_seq.shape[0]):
            spike_t = super().forward(x_seq[t])
            spike_seq.append(spike_t.unsqueeze(0))
            v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, dim=0)
        self.v_seq = torch.cat(v_seq, dim=0)
        return spike_seq

    def reset(self):
        super().reset()
        self.v_seq = None


# ============================================================================
# KEEP_LEVELS UTILITIES
# ============================================================================

KeepLevels = Union[str, Iterable[int]]


def parse_keep_levels(keep_levels_str: str) -> KeepLevels:
    """
    Parse keep_levels from command-line string.
    
    Examples:
        "all" -> "all"
        "detail" -> "detail"
        "0,1" -> [0, 1]
    """
    if keep_levels_str in ["all", "detail", "approx", "finest"]:
        return keep_levels_str
    else:
        try:
            return [int(x.strip()) for x in keep_levels_str.split(",")]
        except ValueError:
            raise ValueError(f"Invalid keep_levels: {keep_levels_str}")


# ============================================================================
# DWT FORWARD/INVERSE WITH TERNARY SPIKING
# ============================================================================

class DWTForwardTernary(nn.Module):
    """
    Forward DWT with ternary spiking after transform.
    Uses pytorch_wavelets library.
    Returns concatenated coefficients and their individual lengths.
    """
    def __init__(
        self,
        wavelet: str = 'db4',
        J: int = 1,
        spike: bool = True,
        vth: float = 1.0,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.J = J
        self.dwt = DWT1DForward(wave=wavelet, J=J, mode='zero')
        
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth),
            neg_v_threshold=float(vth),
            v_reset=0.0,
            detach_reset=False,
        ) if spike else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: [T, B, pd, D, pn]
        returns: (coeffs, coeff_lengths)
            coeffs: [T, B, pd, D, total_len]
            coeff_lengths: list of lengths [len_approx, len_d1, len_d2, ...]
        """
        T, B, pd, D, pn = x.shape
        
        # Reshape for DWT: [T*B*pd*D, 1, pn]
        x_flat = x.reshape(T * B * pd * D, 1, pn)
        
        # Apply DWT
        yl, yh = self.dwt(x_flat)  # yl: [N, 1, len_a], yh: list of [N, 1, len_d]
        
        # Record lengths
        coeff_lengths = [yl.shape[-1]] + [h.shape[-1] for h in yh]
        
        # Concatenate: [approx, detail_1 (coarsest), ..., detail_J (finest)]
        coeffs = [yl]
        for h in yh:
            coeffs.append(h)
        coeffs = torch.cat(coeffs, dim=-1)  # [N, 1, total_len]
        
        # Reshape back: [T, B, pd, D, total_len]
        total_len = coeffs.shape[-1]
        coeffs = coeffs.view(T, B, pd, D, total_len)
        
        # Apply ternary spiking
        return self.neuron(coeffs), coeff_lengths


class DWTInverseTernary(nn.Module):
    """
    Inverse DWT with ternary spiking after reconstruction.
    """
    def __init__(
        self,
        wavelet: str = 'db4',
        J: int = 1,
        spike: bool = True,
        vth: float = 1.0,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.J = J
        self.idwt = DWT1DInverse(wave=wavelet, mode='zero')
        
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth),
            neg_v_threshold=float(vth),
            v_reset=0.0,
            detach_reset=False,
        ) if spike else nn.Identity()

    def forward(self, coeffs: torch.Tensor, coeff_lengths: List[int]) -> torch.Tensor:
        """
        coeffs: [T, B, pd, D, total_len]
        coeff_lengths: list of lengths [len_approx, len_d1, len_d2, ...]
        returns: [T, B, pd, D, reconstructed_len]
        """
        T, B, pd, D, total_len = coeffs.shape
        
        # Reshape: [T*B*pd*D, 1, total_len]
        coeffs_flat = coeffs.reshape(T * B * pd * D, 1, total_len)
        
        # Split into approx and details using actual lengths
        idx = 0
        yl = coeffs_flat[:, :, idx:idx + coeff_lengths[0]]
        idx += coeff_lengths[0]
        
        yh = []
        for j in range(self.J):
            yh.append(coeffs_flat[:, :, idx:idx + coeff_lengths[j + 1]])
            idx += coeff_lengths[j + 1]
        
        # Apply inverse DWT
        x_recon = self.idwt((yl, yh))  # [N, 1, recon_len]
        
        # Reshape back: [T, B, pd, D, recon_len]
        recon_len = x_recon.shape[-1]
        x_recon = x_recon.view(T, B, pd, D, recon_len)
        
        # Apply ternary spiking
        return self.neuron(x_recon)


# ============================================================================
# SpikF-style SWS block with DWT
# ============================================================================

class SWS_DWT(nn.Module):
    """
    SpikF-style Spiking Wavelet Selector using pytorch_wavelets DWT.
    """
    def __init__(
        self,
        patch_num: int,
        D: int,
        patch_dim: int,
        tau: float,
        alpha: float,
        wavelet: str = 'db4',
        J: int = 1,
        keep_levels: KeepLevels = "all",
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.wavelet = wavelet
        self.J = J
        self.keep_levels = keep_levels
        
        # DWT modules
        self.dwt_fwd = DWTForwardTernary(
            wavelet=wavelet,
            J=J,
            spike=transform_spike,
            vth=transform_vth,
        )
        self.dwt_inv = DWTInverseTernary(
            wavelet=wavelet,
            J=J,
            spike=transform_spike,
            vth=transform_vth,
        )
        
        # Will be built dynamically
        self._coeff_lengths = None
        self._total_coeff_len = None
        self._selector_built = False
        self.time2wave = None
        self.level_mask = None
        
        # Same convs as SpikF
        self.intra_conv = nn.Conv2d(
            in_channels=patch_dim, out_channels=patch_dim,
            kernel_size=[5, 1], stride=[1, 1], padding=[2, 0]
        )
        self.inter_conv = nn.Conv2d(
            in_channels=patch_dim, out_channels=patch_dim,
            kernel_size=[3, 1], stride=[1, 1], padding=[1, 0]
        )

        # Spiking neurons
        self.generator_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch", v_threshold=0.1)
        self.mp_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.sws_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.inter_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")

        # Batch norms
        self.bn1 = None  # Will be created dynamically
        self.bn2 = nn.BatchNorm2d(patch_dim)
        self.bn3 = nn.BatchNorm2d(patch_dim)
        self.bn4 = nn.BatchNorm2d(patch_dim)
        
        self.patch_dim = patch_dim

    def _make_level_mask(self, coeff_lengths: List[int], keep_levels: KeepLevels) -> torch.Tensor:
        """
        Create mask for DWT coefficients with variable-length subbands.
        
        coeff_lengths: [len_approx, len_d1, len_d2, ...] where d1 is coarsest
        keep_levels:
          - "all": keep everything
          - "detail": keep all detail (drop approx)
          - "approx": keep only approx
          - "finest": keep only finest detail
          - list of ints: 0=approx, 1=coarsest detail, ..., J=finest detail
        """
        total_len = sum(coeff_lengths)
        mask = torch.zeros(total_len, dtype=torch.float32)
        
        # Build index ranges for each level
        level_ranges = []
        idx = 0
        for length in coeff_lengths:
            level_ranges.append((idx, idx + length))
            idx += length
        
        J = len(coeff_lengths) - 1  # Number of detail levels
        
        if isinstance(keep_levels, str):
            mode = keep_levels.lower()
            if mode == "all":
                mask[:] = 1.0
            elif mode == "detail":
                # Keep all detail coefficients (skip approx at index 0)
                for i in range(1, len(level_ranges)):
                    start, end = level_ranges[i]
                    mask[start:end] = 1.0
            elif mode == "approx":
                # Keep only approximation coefficients
                start, end = level_ranges[0]
                mask[start:end] = 1.0
            elif mode == "finest":
                # Keep only finest detail (last level)
                start, end = level_ranges[-1]
                mask[start:end] = 1.0
            else:
                raise ValueError(f"Unknown keep_levels='{keep_levels}'")
        else:
            # List of level indices
            for lvl in keep_levels:
                if lvl < 0 or lvl > J:
                    raise ValueError(f"level {lvl} out of range [0..{J}]")
                start, end = level_ranges[lvl]
                mask[start:end] = 1.0
        
        return mask

    def _build_dynamic_layers(self, total_coeff_len: int, coeff_lengths: List[int], device: torch.device):
        """Build layers that depend on DWT output size."""
        if self._selector_built and self._total_coeff_len == total_coeff_len:
            return
            
        self._coeff_lengths = coeff_lengths
        self._total_coeff_len = total_coeff_len
        
        # Create selector linear layer
        self.time2wave = nn.Linear(self.patch_num, total_coeff_len).to(device)
        
        # Create level mask using actual coefficient lengths
        mask = self._make_level_mask(coeff_lengths, self.keep_levels)
        self.level_mask = mask.view(1, 1, 1, 1, total_coeff_len).to(device)
        
        # Create bn1
        self.bn1 = nn.BatchNorm2d(self.patch_dim).to(device)
        
        self._selector_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, pd, pn, D]
        returns: [T, B, pd, pn, D]
        """
        res_x = x
        T, B, pd, pn, D = x.shape
        assert pn == self.patch_num, f"Expected pn={self.patch_num}, got {pn}"

        # [T, B, pd, D, pn]
        x_t = x.transpose(-1, -2).contiguous()

        # Forward DWT - now returns coefficients AND their lengths
        wav, coeff_lengths = self.dwt_fwd(x_t)  # wav: [T, B, pd, D, total_coeff_len]
        total_coeff_len = wav.shape[-1]
        
        # Build dynamic layers if needed (using actual coeff_lengths)
        self._build_dynamic_layers(total_coeff_len, coeff_lengths, x.device)

        # SpikF-style selector
        selector = self.time2wave(x_t)          # [T, B, pd, D, total_coeff_len]
        selector = selector.flatten(0, 1)       # [T*B, pd, D, total_coeff_len]
        selector = self.bn1(selector)
        selector = selector.view(T, B, pd, D, total_coeff_len)
        selector = self.generator_lif(selector)
        selector = selector.sum(dim=0, keepdim=True)     # [1, B, pd, D, total_coeff_len]
        selector = self.mp_lif(selector)
        selector = selector.repeat(T, 1, 1, 1, 1).float()  # [T, B, pd, D, total_coeff_len]

        # Apply level mask
        selector = selector * self.level_mask
        wav = wav * self.level_mask

        # Select wavelet coefficients
        remain = selector * wav  # [T, B, pd, D, total_coeff_len]

        # Inverse DWT - pass the coefficient lengths
        current = self.dwt_inv(remain, coeff_lengths)  # [T, B, pd, D, recon_len]
        
        # Handle size mismatch (DWT may change size slightly)
        recon_len = current.shape[-1]
        if recon_len > pn:
            current = current[..., :pn]
        elif recon_len < pn:
            current = nn.functional.pad(current, (0, pn - recon_len))

        # [T, B, pd, pn, D]
        current = current.transpose(-1, -2).contiguous()
        current = current.flatten(0, 1)        # [T*B, pd, pn, D]
        current = self.bn2(current)
        current = current.view(T, B, pd, pn, D)

        spike = self.sws_lif(current)
        x = spike + res_x
        res_x = x

        # Intra conv branch
        y = x.flatten(0, 1)        # [T*B, pd, pn, D]
        y = self.intra_conv(y)
        y = self.bn3(y)
        y = y.view(T, B, pd, pn, D)
        y = self.intra_lif(y) + res_x
        res_x = y

        # Inter conv branch
        z = y.transpose(0, 3).contiguous()     # [pn, B, pd, T, D]
        z = z.flatten(0, 1)                    # [pn*B, pd, T, D]
        z = self.inter_conv(z)
        z = self.bn4(z)
        z = z.view(pn, B, pd, T, D).transpose(0, 3).contiguous()  # [T, B, pd, pn, D]
        z = self.inter_lif(z)
        z = z + res_x

        return z


# ============================================================================
# SPE (from SpikF)
# ============================================================================

class SPE(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, tau, D):
        super().__init__()
        self.patch_projector = nn.Linear(input_len // patch_num, patch_dim)
        self.bn = nn.BatchNorm2d(patch_dim)
        self.encoder_lif = MultiStepLIFNode(tau=tau, detach_reset=False, backend='torch')

        self.D = D
        self.T = T
        self.patch_dim = patch_dim
        self.patch_num = patch_num

    def forward(self, x):
        B, L, D = x.shape

        x = x.view(B, self.patch_num, L // self.patch_num, D).contiguous()
        x = x.transpose(-1, -2).contiguous()
        x = self.patch_projector(x)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.view(self.T, B, self.patch_dim, self.patch_num, D)
        x = self.encoder_lif(x)

        return x


# ============================================================================
# Main model: SwaveNetV1_2
# ============================================================================

class SwaveNetV1_2(nn.Module):
    def __init__(
        self,
        input_len: int,
        patch_num: int,
        patch_dim: int,
        T: int,
        blocks: int,
        D: int,
        pred_len: int,
        tau: float,
        alpha: float,
        hidden_dim: int,
        wavelet: str = 'db4',
        J: int = 1,
        keep_levels: KeepLevels = "all",
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        
        self.T = T
        self.pred_len = pred_len
        self.patch_num = patch_num
        self.wavelet = wavelet
        self.J = J
        self.keep_levels = keep_levels
        
        print(f"[SwaveNetV1_2] wavelet={wavelet}, J={J}, keep_levels={keep_levels}, patch_num={patch_num}")
        
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)

        self.SWSs = nn.ModuleList([
            SWS_DWT(
                patch_num=patch_num,
                D=D,
                patch_dim=patch_dim,
                tau=tau,
                alpha=alpha,
                wavelet=wavelet,
                J=J,
                keep_levels=keep_levels,
                transform_spike=transform_spike,
                transform_vth=transform_vth,
            )
            for _ in range(blocks)
        ])

        self.dense1 = nn.Linear(patch_num * patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        returns: [T, B, pred_len, D]
        """
        # Instance normalization
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        x = self.SPE(x)  # [T, B, pd, pn, D]
        T, B, pd, pn, D = x.shape

        for blk in self.SWSs:
            x = blk(x)

        # Decode (same as SpikF)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [T, B, D, pd, pn]
        x = x.flatten(-2, -1)                      # [T, B, D, pd*pn]
        x = self.dense1(x)                         # [T, B, D, hidden]
        x = x.flatten(0, 1)                        # [T*B, D, hidden]
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)                         # [T*B, D, pred_len]
        x = x.transpose(-1, -2).contiguous()       # [T*B, pred_len, D]
        x = x.view(T, B, self.pred_len, D)

        # De-normalize
        x = x * std
        x = x + mean.repeat(T, 1, 1, 1)
        return x