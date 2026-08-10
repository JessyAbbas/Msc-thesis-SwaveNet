# SwaveNetV1_2_LightSpikformer 
# DWT via pytorch_wavelets (db4) with ternary spiking
# SpikF-style frequency selector 
# Intra-conv: Same
# Inter-patch: REPLACED with Light Spikformer block:
# XNOR-based SSA with Gray-PE + Log-PE (from SeqSNN repo)
# DWC (Depth-Wise Conv) for high-freq restoration (from MaxFormer repo)

# This is SwaveNetV1_LightSpikformer but with pytorch_wavelets DWT instead of Haar matrix

import math
from abc import abstractmethod
from typing import Callable, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import surrogate, base
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# Type alias

KeepLevels = Union[str, List[int]]


def parse_keep_levels(keep_levels_str: str) -> KeepLevels:
    if keep_levels_str in ["all", "detail", "approx", "finest"]:
        return keep_levels_str
    else:
        try:
            return [int(x.strip()) for x in keep_levels_str.split(",")]
        except ValueError:
            raise ValueError(f"Invalid keep_levels: {keep_levels_str}")


# TERNARY SPIKING NEURONS (from SWformer, same as SwaveNetV1)

class NegBaseNode(base.MemoryModule):
    def __init__(self, v_threshold=1.0, neg_v_threshold=1.0, v_reset=0.0,
                 surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        super().__init__()
        assert isinstance(v_threshold, float)
        assert isinstance(neg_v_threshold, float)
        assert isinstance(v_reset, float) or v_reset is None
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
    def neuronal_charge(self, x):
        raise NotImplementedError

    def neuronal_fire(self):
        pos_spike = self.surrogate_function(self.v - self.v_threshold)
        neg_spike = self.surrogate_function(-(self.v + self.neg_v_threshold + self.inhibit))
        return pos_spike - neg_spike

    def neuronal_reset(self, spike):
        spike_d = spike.detach() if self.detach_reset else spike
        pos = spike_d > 0
        neg = spike_d < 0
        spike_d_vth = pos * self.v_threshold + neg * self.neg_v_threshold
        if self.v_reset is None:
            self.v = self.v - spike_d_vth
        else:
            self.v = (1.0 - spike_d.abs()) * self.v + spike_d * self.v_reset

    def forward(self, x):
        self.neuronal_charge(x)
        s = self.neuronal_fire()
        self.neuronal_reset(s)
        return s


class NegIFNode(NegBaseNode):
    def __init__(self, v_threshold=1.0, neg_v_threshold=1.0, v_reset=0.0,
                 surrogate_function=surrogate.Sigmoid(), detach_reset=False):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x):
        self.v = self.v + x


class MultiStepNegIFNode(NegIFNode):
    def forward(self, x):
        T = x.shape[0]
        outputs = []
        for t in range(T):
            outputs.append(super().forward(x[t]))
        return torch.stack(outputs, dim=0)


# ============================================================================
# DWT FORWARD/INVERSE WITH TERNARY SPIKING (from SwaveNetV1_2)
# Replaces the Haar matrix approach
# ============================================================================

class DWTForwardTernary(nn.Module):
    def __init__(self, wavelet='db4', J=1, spike=True, vth=1.0):
        super().__init__()
        self.wavelet = wavelet
        self.J = J
        self.dwt = DWT1DForward(wave=wavelet, J=J, mode='zero')
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth), neg_v_threshold=float(vth),
            v_reset=0.0, detach_reset=False,
        ) if spike else nn.Identity()

    def forward(self, x):
        """
        x: [T, B, pd, D, pn]
        returns: (coeffs [T, B, pd, D, total_len], coeff_lengths)
        """
        T, B, pd, D, pn = x.shape
        x_flat = x.reshape(T * B * pd * D, 1, pn)
        yl, yh = self.dwt(x_flat)
        coeff_lengths = [yl.shape[-1]] + [h.shape[-1] for h in yh]
        coeffs = torch.cat([yl] + list(yh), dim=-1)
        total_len = coeffs.shape[-1]
        coeffs = coeffs.view(T, B, pd, D, total_len)
        return self.neuron(coeffs), coeff_lengths


class DWTInverseTernary(nn.Module):
    def __init__(self, wavelet='db4', J=1, spike=True, vth=1.0):
        super().__init__()
        self.wavelet = wavelet
        self.J = J
        self.idwt = DWT1DInverse(wave=wavelet, mode='zero')
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth), neg_v_threshold=float(vth),
            v_reset=0.0, detach_reset=False,
        ) if spike else nn.Identity()

    def forward(self, coeffs, coeff_lengths):
        """
        coeffs: [T, B, pd, D, total_len]
        coeff_lengths: [len_approx, len_d1, ..., len_dJ]
        returns: [T, B, pd, D, recon_len]
        """
        T, B, pd, D, total_len = coeffs.shape
        coeffs_flat = coeffs.reshape(T * B * pd * D, 1, total_len)
        idx = 0
        yl = coeffs_flat[:, :, idx:idx + coeff_lengths[0]]
        idx += coeff_lengths[0]
        yh = []
        for j in range(self.J):
            yh.append(coeffs_flat[:, :, idx:idx + coeff_lengths[j + 1]])
            idx += coeff_lengths[j + 1]
        x_recon = self.idwt((yl, yh))
        recon_len = x_recon.shape[-1]
        x_recon = x_recon.view(T, B, pd, D, recon_len)
        return self.neuron(x_recon)


# ============================================================================
# LEVEL MASK FOR DWT (variable-length subbands, replaces Haar mask)
# ============================================================================

def make_dwt_level_mask(coeff_lengths, keep_levels):
    """
    Create mask for DWT coefficients with variable-length subbands.
    coeff_lengths: [len_approx, len_d1, len_d2, ...] where d1 is coarsest
    """
    total_len = sum(coeff_lengths)
    mask = torch.zeros(total_len, dtype=torch.float32)
    level_ranges = []
    idx = 0
    for length in coeff_lengths:
        level_ranges.append((idx, idx + length))
        idx += length
    J = len(coeff_lengths) - 1

    if isinstance(keep_levels, str):
        mode = keep_levels.lower()
        if mode == "all":
            mask[:] = 1.0
        elif mode == "detail":
            for i in range(1, len(level_ranges)):
                start, end = level_ranges[i]
                mask[start:end] = 1.0
        elif mode == "approx":
            start, end = level_ranges[0]
            mask[start:end] = 1.0
        elif mode == "finest":
            start, end = level_ranges[-1]
            mask[start:end] = 1.0
        else:
            raise ValueError(f"Unknown keep_levels='{keep_levels}'")
    else:
        for lvl in keep_levels:
            if lvl < 0 or lvl > J:
                raise ValueError(f"level {lvl} out of range [0..{J}]")
            start, end = level_ranges[lvl]
            mask[start:end] = 1.0
    return mask


# ============================================================================
# GRAY-PE (EXACT from spikformer_xnor_gray.py)
# ============================================================================

def generate_gray_code_matrix(M, num_bits=10):
    """
    Exact implementation from SeqSNN/spikformer_xnor_gray.py.
    
    M: [T, B, H, L, D//H] - the Q or K tensor after multi-head reshape
    Returns: M concatenated with gray code along last dim -> [T, B, H, L, D//H + num_bits]
    
    Key detail: indices = torch.arange(T * L), so gray codes span T*L positions,
    then reshaped to [T, 1, 1, L, num_bits].
    """
    T, B, H, L, _ = M.shape
    
    indices = torch.arange(T * L)
    gray_codes = indices ^ (indices >> 1)
    
    gray_code_matrix = ((gray_codes.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1)) & 1).float()
    gray_code_matrix = gray_code_matrix.view(T, 1, 1, L, num_bits)
    gray_code_matrix = gray_code_matrix.expand(T, B, H, L, num_bits)
    
    result = torch.cat((M, gray_code_matrix.to(M.device)), dim=-1)
    return result

# ============================================================================
# LOG-PE (EXACT from spikformer_xnor_log.py)
# ============================================================================

def create_symmetric_matrix(L):
    """
    Exact implementation from SeqSNN/spikformer_xnor_log.py.
    
    Creates a symmetric L×L matrix where:
    - Diagonal = ceil(log2(L))
    - Off-diagonal: linspace decay from diag_value-1 to 0
    """
    diag_value = math.ceil(math.log2(L))
    matrix = torch.zeros((L, L), dtype=torch.int)
    for i in range(L):
        matrix[i, i] = diag_value
        if i < L - 1:
            values = torch.linspace(diag_value - 1, 0, steps=L - i - 1).round().int()
            matrix[i, i + 1:] = values
            matrix[i + 1:, i] = values
    return matrix


# ============================================================================
# XNOR SSA with Gray-PE + Log-PE (adapted from SeqSNN for 1D time-series)
# ============================================================================

class XNOR_SSA_GrayLog(nn.Module):
    """
    XNOR-based Spiking Self-Attention with Gray-PE and Log-PE.
    
    Follows the EXACT patterns from:
    - spikformer_xnor.py: XNOR attention = D - sum_q - sum_k + 2*qk_matmul
    - spikformer_xnor_gray.py: Gray code concat to Q,K
    - spikformer_xnor_log.py: Symmetric matrix added to attn map
    
    Adapted for 1D time-series patches (not 2D image patches).
    """
    
    def __init__(self, dim, num_heads=8, tau=2.0, common_thr=1.0,
                 use_gray_pe=True, use_log_pe=True, gray_num_bits=10):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.use_gray_pe = use_gray_pe
        self.use_log_pe = use_log_pe
        self.gray_num_bits = gray_num_bits
        
        # Learnable scale (from actual code: init=-4.0 for Gray/Log variants)
        self.scale = nn.Parameter(torch.tensor(-4.0), requires_grad=True)
        
        # Q, K, V projections (using Linear like SeqSNN, not Conv1d)
        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        
        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        
        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        
        # Attention LIF (v_threshold = common_thr/2, from actual code)
        self.attn_lif = MultiStepLIFNode(tau=tau, v_threshold=common_thr / 2,
                                          detach_reset=True, backend="torch")
        
        # Output projection
        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
    
    def forward(self, x):
        """
        x: [T, B, L, D] (T=spiking timesteps, B=batch, L=patches, D=dim)
        returns: [T, B, L, D]
        """
        T, B, L, D = x.shape
        
        # Q, K, V projections (exact pattern from SeqSNN)
        x_for_qkv = x.flatten(0, 1)  # [TB, L, D]
        
        q_m_out = self.q_m(x_for_qkv)  # [TB, L, D]
        q_m_out = self.q_bn(q_m_out.transpose(-1, -2)).transpose(-1, -2)
        q_m_out = q_m_out.reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.num_heads, D // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()  # [T, B, H, L, D//H]
        
        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_bn(k_m_out.transpose(-1, -2)).transpose(-1, -2)
        k_m_out = k_m_out.reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.num_heads, D // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4).contiguous()  # [T, B, H, L, D//H]
        
        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_bn(v_m_out.transpose(-1, -2)).transpose(-1, -2)
        v_m_out = v_m_out.reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.num_heads, D // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4).contiguous()  # [T, B, H, L, D//H]
        
        #  Compute XNOR attention with optional PE 
        if self.use_gray_pe:
            # Gray-PE: concatenate gray codes (exact from spikformer_xnor_gray.py)
            q_aug = generate_gray_code_matrix(q, num_bits=self.gray_num_bits)  # [T,B,H,L,D'+bits]
            k_aug = generate_gray_code_matrix(k, num_bits=self.gray_num_bits)  # [T,B,H,L,D'+bits]
            
            # Memory-efficient XNOR (exact from source)
            D_new = q_aug.size(-1)
            sum_q = q_aug.sum(dim=-1).unsqueeze(-1)  # [T,B,H,L,1]
            sum_k = k_aug.sum(dim=-1).unsqueeze(-2)  # [T,B,H,1,L]
            qk_matmul = torch.matmul(q_aug, k_aug.transpose(-1, -2))  # [T,B,H,L,L]
            attn = D_new - sum_q - sum_k + 2 * qk_matmul  # [T, B, H, L, L]
        else:
            # Plain XNOR without Gray-PE (from spikformer_xnor.py)
            D_new = q.size(-1)
            sum_q = q.sum(dim=-1)
            sum_k = k.sum(dim=-1)
            qk_matmul = torch.matmul(q, k.transpose(-1, -2))
            sum_q = sum_q.unsqueeze(-1)
            sum_k = sum_k.unsqueeze(-2)
            attn = D_new - sum_q - sum_k + 2 * qk_matmul  # [T, B, H, L, L]
        
        # Apply Log-PE or scale
        if self.use_log_pe:
            # Log-PE: add symmetric matrix (exact from spikformer_xnor_log.py)
            tmp = create_symmetric_matrix(L).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attn = (attn + tmp.to(attn.device)) * torch.sigmoid(self.scale)
        else:
            attn = attn * torch.sigmoid(self.scale)
        
        # Apply attention to V 
        x = attn @ v  # [T, B, H, L, D//H]
        
        # Output projection (exact pattern from SeqSNN)
        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)  # [TB, L, D]
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        
        return x


# ============================================================================
# DWC MIXER (EXACT pattern from MaxFormer mixer_hub.py Mixer_DWC5)
# Adapted from 2D (H,W) to 1D (N patches) for time-series
# ============================================================================

class SpikingDWC(nn.Module):
    """
    Spiking Depth-Wise Convolution - adapted from MaxFormer Mixer_DWC3/DWC5.
    
    Original (2D images): LIF -> Conv2d(groups=dim) -> BN2d -> + identity
    Adapted (1D patches): LIF -> Conv1d(groups=dim) -> BN1d -> + identity
    """
    
    def __init__(self, dim, tau=2.0, kernel_size=5):
        super().__init__()
        # Exact pattern from Mixer_DWC: neuron -> conv -> bn -> + identity
        self.conv_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv_bn = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        """x: [T, B, N, D] -> [T, B, N, D]"""
        T, B, N, D = x.shape
        identity = x
        
        # LIF -> reshape -> DWC -> BN -> reshape -> + identity
        x = self.conv_neuron(x)  # [T, B, N, D]
        x = x.reshape(T * B, N, D).permute(0, 2, 1)  # [TB, D, N]
        x = self.conv(x)
        x = self.conv_bn(x)
        x = x.permute(0, 2, 1).reshape(T, B, N, D).contiguous()
        
        x = x + identity
        return x


# ============================================================================
# LIGHT SPIKFORMER BLOCK (XNOR-SSA + DWC)
# ============================================================================

class LightSpikformerBlock(nn.Module):
    """
    Combines XNOR-SSA (with Gray-PE + Log-PE) and DWC.
    Pattern follows MaxFormer SSA_DWC: SSA first, then DWC for high-freq.
    """
    
    def __init__(self, dim, num_heads=8, tau=2.0, common_thr=1.0,
                 use_gray_pe=True, use_log_pe=True,
                 gray_num_bits=10, dwc_kernel_size=5):
        super().__init__()
        
        self.attn = XNOR_SSA_GrayLog(
            dim=dim, num_heads=num_heads, tau=tau, common_thr=common_thr,
            use_gray_pe=use_gray_pe, use_log_pe=use_log_pe,
            gray_num_bits=gray_num_bits,
        )
        self.dwc = SpikingDWC(dim=dim, tau=tau, kernel_size=dwc_kernel_size)
    
    def forward(self, x):
        """x: [T, B, N, D] -> [T, B, N, D]"""
        x = x + self.attn(x)  # residual attention
        x = self.dwc(x)       # DWC with built-in residual
        return x


# ============================================================================
# SWS Block with DWT + Light Spikformer
# ============================================================================

class SWS_DWT_LightSpikformer(nn.Module):
    """
    Spiking Wavelet Selector with Light Spikformer replacing inter-conv.
    Uses pytorch_wavelets DWT instead of Haar matrix.
    
    1. Wavelet Branch: DWT -> SpikF selector -> Inverse DWT
    2. Intra-conv: SAME as SwaveNetV1
    3. Light Spikformer: REPLACES inter-conv
    """
    
    def __init__(self, patch_num, D, patch_dim, tau, alpha,
                 wavelet='db4', J=1,
                 num_heads=8, keep_levels="all",
                 transform_spike=True, transform_vth=1.0,
                 use_gray_pe=True, use_log_pe=True,
                 gray_num_bits=10, dwc_kernel_size=5):
        super().__init__()

        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.D = D
        self.wavelet = wavelet
        self.J = J
        self.keep_levels = keep_levels

        # DWT modules (replaces HaarForward/HaarInverse)
        self.dwt_fwd = DWTForwardTernary(wavelet=wavelet, J=J,
                                          spike=transform_spike, vth=transform_vth)
        self.dwt_inv = DWTInverseTernary(wavelet=wavelet, J=J,
                                          spike=transform_spike, vth=transform_vth)

        # Dynamic layers (DWT output size depends on wavelet/J/input)
        self._coeff_lengths = None
        self._total_coeff_len = None
        self._selector_built = False
        self.time2wave = None
        self.level_mask = None

        # Intra conv (SAME as SwaveNetV1)
        self.intra_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[5, 1], stride=[1, 1], padding=[2, 0])

        # Light Spikformer (REPLACES inter-conv)
        self.light_spikformer = LightSpikformerBlock(
            dim=patch_dim, num_heads=num_heads, tau=tau,
            use_gray_pe=use_gray_pe, use_log_pe=use_log_pe,
            gray_num_bits=gray_num_bits, dwc_kernel_size=dwc_kernel_size,
        )

        # Spiking neurons
        self.generator_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch", v_threshold=0.1)
        self.mp_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.sws_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")

        # Batch norms
        self.bn1 = None  # Built dynamically
        self.bn2 = nn.BatchNorm2d(patch_dim)
        self.bn3 = nn.BatchNorm2d(patch_dim)

    def _build_dynamic_layers(self, total_coeff_len, coeff_lengths, device):
        """Build layers that depend on DWT output size (called once)."""
        if self._selector_built and self._total_coeff_len == total_coeff_len:
            return
        self._coeff_lengths = coeff_lengths
        self._total_coeff_len = total_coeff_len

        self.time2wave = nn.Linear(self.patch_num, total_coeff_len).to(device)
        mask = make_dwt_level_mask(coeff_lengths, self.keep_levels)
        self.level_mask = mask.view(1, 1, 1, 1, total_coeff_len).to(device)
        self.bn1 = nn.BatchNorm2d(self.patch_dim).to(device)
        self._selector_built = True

    def forward(self, x):
        """x: [T, B, pd, pn, D] -> [T, B, pd, pn, D]"""
        res_x = x
        T, B, pd, pn, D = x.shape

        # ========== Wavelet Branch (DWT replaces Haar) ==========
        x_t = x.transpose(-1, -2).contiguous()  # [T, B, pd, D, pn]
        wav, coeff_lengths = self.dwt_fwd(x_t)
        total_coeff_len = wav.shape[-1]

        self._build_dynamic_layers(total_coeff_len, coeff_lengths, x.device)

        # SpikF-style selector
        selector = self.time2wave(x_t)
        selector = selector.flatten(0, 1)
        selector = self.bn1(selector)
        selector = selector.view(T, B, pd, D, total_coeff_len)
        selector = self.generator_lif(selector)
        selector = selector.sum(dim=0, keepdim=True)
        selector = self.mp_lif(selector)
        selector = selector.repeat(T, 1, 1, 1, 1).float()

        selector = selector * self.level_mask
        wav = wav * self.level_mask
        remain = selector * wav

        current = self.dwt_inv(remain, coeff_lengths)

        # Handle size mismatch (DWT may change size slightly)
        recon_len = current.shape[-1]
        if recon_len > pn:
            current = current[..., :pn]
        elif recon_len < pn:
            current = F.pad(current, (0, pn - recon_len))

        current = current.transpose(-1, -2).contiguous()
        current = current.flatten(0, 1)
        current = self.bn2(current)
        current = current.view(T, B, pd, pn, D)
        spike = self.sws_lif(current)
        x = spike + res_x
        res_x = x

        # ========== Intra-conv (SAME as SwaveNetV1) ==========
        y = x.flatten(0, 1)
        y = self.intra_conv(y)
        y = self.bn3(y)
        y = y.view(T, B, pd, pn, D)
        y = self.intra_lif(y) + res_x
        res_x = y

        # ========== Light Spikformer (REPLACES inter-conv) ==========
        z = y.permute(0, 1, 4, 3, 2).contiguous()  # [T, B, D, pn, pd]
        T_, B_, D_feat, pn_, pd_ = z.shape
        z = z.reshape(T_, B_ * D_feat, pn_, pd_)    # [T, B*D, pn, pd]
        z = self.light_spikformer(z)                  # [T, B*D, pn, pd]
        z = z.reshape(T_, B_, D_feat, pn_, pd_)
        z = z.permute(0, 1, 4, 3, 2).contiguous()   # [T, B, pd, pn, D]
        z = z + res_x

        return z


# ============================================================================
# SPE (same as SwaveNetV1)
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
# Main Model
# ============================================================================

class SwaveNetV1_2_LightSpikformer(nn.Module):
    """
    Spiking Wavelet Network V1.2 with Light Spikformer.
    Uses pytorch_wavelets DWT (db4/db8/sym4/sym8) instead of Haar matrix.
    
    Changes from SwaveNetV1_2:
    - Inter-conv: REPLACED by Light Spikformer (XNOR SSA + Gray-PE + Log-PE + DWC)
    """
    
    def __init__(self, input_len, patch_num, patch_dim, T, blocks, D, pred_len,
                 tau, alpha, hidden_dim, wavelet='db4', J=1,
                 num_heads=8, keep_levels="all",
                 transform_spike=True, transform_vth=1.0,
                 use_gray_pe=True, use_log_pe=True,
                 gray_num_bits=10, dwc_kernel_size=5):
        super().__init__()
        self.T = T
        self.pred_len = pred_len
        self.patch_num = patch_num

        pe_parts = []
        if use_gray_pe: pe_parts.append("Gray-PE")
        if use_log_pe: pe_parts.append("Log-PE")
        print(f"[SwaveNetV1_2_LightSpikformer] wavelet={wavelet}, J={J}, "
              f"patch_num={patch_num}, patch_dim={patch_dim}, "
              f"heads={num_heads}, PE={'+'.join(pe_parts) or 'None'}, "
              f"DWC_k={dwc_kernel_size}, keep_levels={keep_levels}")

        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)
        self.SWSs = nn.ModuleList([
            SWS_DWT_LightSpikformer(
                patch_num=patch_num, D=D, patch_dim=patch_dim, tau=tau, alpha=alpha,
                wavelet=wavelet, J=J, num_heads=num_heads, keep_levels=keep_levels,
                transform_spike=transform_spike, transform_vth=transform_vth,
                use_gray_pe=use_gray_pe, use_log_pe=use_log_pe,
                gray_num_bits=gray_num_bits, dwc_kernel_size=dwc_kernel_size,
            )
            for _ in range(blocks)
        ])
        self.dense1 = nn.Linear(patch_num * patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = nn.GELU()

    def forward(self, x):
        """x: [B, L, D] -> [T, B, pred_len, D]"""
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        x = self.SPE(x)
        T, B, pd, pn, D = x.shape
        for blk in self.SWSs:
            x = blk(x)

        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(-2, -1)
        x = self.dense1(x)
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(T, B, self.pred_len, D)

        x = x * std
        x = x + mean.repeat(T, 1, 1, 1)
        return x