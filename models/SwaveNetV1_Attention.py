# SwaveNetV1_Attention: Spiking Wavelet Network with Spike-Driven Self-Attention (SDSA)
# SDSA is always enabled - this is SwaveNetV1 + SDSA integrated

import math
from abc import abstractmethod
from typing import Callable, Iterable, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.clock_driven import surrogate, base
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


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
        return pos_spike - neg_spike

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
# HAAR WAVELET UTILITIES
# ============================================================================

def _is_power_of_two(n: int) -> bool:
    return (n > 0) and (n & (n - 1) == 0)

def haar_1d_matrix_np(n: int) -> np.ndarray:
    if not _is_power_of_two(n):
        raise ValueError(f"Haar size must be power of 2, got n={n}")
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    H_next = haar_1d_matrix_np(n // 2)
    upper = np.kron(H_next, [1.0, 1.0]).astype(np.float32)
    lower = np.kron(np.eye(len(H_next), dtype=np.float32), [1.0, -1.0]).astype(np.float32)
    H = np.vstack((upper, lower)).astype(np.float32)
    return H

def haar_matrix(n: int, device: torch.device) -> torch.Tensor:
    H = torch.from_numpy(haar_1d_matrix_np(n)).to(device=device, dtype=torch.float32)
    norms = torch.linalg.norm(H, dim=1, keepdim=True)
    H = H / (norms + 1e-12)
    return H

def haar_level_slices(n: int) -> List[slice]:
    if not _is_power_of_two(n):
        raise ValueError(f"n must be power of 2, got {n}")
    J = int(math.log2(n))
    out = [slice(0, 1)]
    for l in range(1, J + 1):
        out.append(slice(2 ** (l - 1), 2 ** l))
    return out

KeepLevels = Union[str, Iterable[int]]

def make_haar_level_mask(n: int, keep_levels: KeepLevels) -> torch.Tensor:
    slices = haar_level_slices(n)
    J = len(slices) - 1
    mask = torch.zeros(n, dtype=torch.float32)

    if isinstance(keep_levels, str):
        mode = keep_levels.lower()
        if mode == "all":
            mask[:] = 1.0
        elif mode == "detail":
            mask[slices[1].start : slices[J].stop] = 1.0
        elif mode == "approx":
            mask[slices[0]] = 1.0
        elif mode == "finest":
            mask[slices[J]] = 1.0
        else:
            raise ValueError(f"Unknown keep_levels='{keep_levels}'")
    else:
        for lvl in keep_levels:
            if lvl < 0 or lvl > J:
                raise ValueError(f"level {lvl} out of range [0..{J}]")
            mask[slices[lvl]] = 1.0
    return mask

def parse_keep_levels(keep_levels_str: str) -> KeepLevels:
    if keep_levels_str in ["all", "detail", "approx", "finest"]:
        return keep_levels_str
    else:
        try:
            return [int(x.strip()) for x in keep_levels_str.split(",")]
        except ValueError:
            raise ValueError(f"Invalid keep_levels: {keep_levels_str}")


# ============================================================================
# HAAR FORWARD/INVERSE WITH TERNARY SPIKING
# ============================================================================

class HaarForward(nn.Module):
    def __init__(self, n: int, spike: bool = True, vth: float = 1.0, detach_reset: bool = False):
        super().__init__()
        self.n = n
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth), neg_v_threshold=float(vth), v_reset=0.0, detach_reset=detach_reset,
        ) if spike else nn.Identity()
        self.register_buffer("H", torch.empty(0), persistent=False)
        self._built = False

    def _build(self, device: torch.device):
        H = haar_matrix(self.n, device)
        self.H = H
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built or self.H.numel() == 0 or self.H.device != x.device:
            self._build(x.device)
        y = torch.matmul(x, self.H.T)
        return self.neuron(y)


class HaarInverse(nn.Module):
    def __init__(self, n: int, spike: bool = True, vth: float = 1.0, detach_reset: bool = False):
        super().__init__()
        self.n = n
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth), neg_v_threshold=float(vth), v_reset=0.0, detach_reset=detach_reset,
        ) if spike else nn.Identity()
        self.register_buffer("H", torch.empty(0), persistent=False)
        self._built = False

    def _build(self, device: torch.device):
        H = haar_matrix(self.n, device)
        self.H = H
        self._built = True

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if not self._built or self.H.numel() == 0 or self.H.device != y.device:
            self._build(y.device)
        x = torch.matmul(y, self.H)
        return self.neuron(x)


# ============================================================================
# SPIKE-DRIVEN SELF-ATTENTION (SDSA) - from Spike-Driven Transformer
# ============================================================================

class SDSA(nn.Module):
    """
    Spike-Driven Self-Attention from Spike-Driven Transformer (NeurIPS 2023).
    
    Key idea:
    - Q, K, V are all binary spikes from LIF neurons
    - Instead of softmax(Q @ K.T) @ V, use:
      kv = K * V (element-wise mask)
      kv = sum(kv, dim=tokens) (sparse addition)
      out = Q * kv (element-wise mask)
    - Only mask and addition, no matrix multiplication!
    - Linear complexity O(N*D) instead of O(N^2*D)
    - 87.2× lower energy than vanilla attention
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        tau: float = 2.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # from original SDSA
        
        # Q, K, V projections
        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)
        
        # Spiking neurons for Q, K, V (all produce binary spikes)
        self.q_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.k_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.v_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        
        # Attention LIF (lower threshold)
        self.attn_lif = MultiStepLIFNode(tau=tau, v_threshold=0.5, detach_reset=True, backend="torch")
        
        # Output projection
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        
        # Input LIF
        self.input_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, N, D] (T=timesteps, B=batch, N=tokens/patches, D=dim)
        returns: [T, B, N, D]
        """
        T, B, N, D = x.shape
        identity = x
        
        # Input spike
        x = self.input_lif(x)
        
        # Q, K, V projections with spiking
        x_flat = x.reshape(T * B * N, D)
        
        q = self.q_linear(x_flat)
        q = self.q_bn(q).reshape(T, B, N, D)
        q = self.q_lif(q)
        
        k = self.k_linear(x_flat)
        k = self.k_bn(k).reshape(T, B, N, D)
        k = self.k_lif(k)
        
        v = self.v_linear(x_flat)
        v = self.v_bn(v).reshape(T, B, N, D)
        v = self.v_lif(v)
        
        # Reshape for multi-head: [T, B, num_heads, N, head_dim]
        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        # ============ SDSA Core: mask and addition only ============
        # Step 1: K * V element-wise (mask operation on binary spikes)
        kv = k.mul(v)  # [T, B, heads, N, head_dim]
        
        # Step 2: Sum over tokens (sparse addition)
        kv = kv.sum(dim=-2, keepdim=True)  # [T, B, heads, 1, head_dim]
        
        # Step 3: Apply LIF to get spike attention
        kv = self.attn_lif(kv)
        
        # Step 4: Q * kv element-wise (mask operation)
        out = q.mul(kv)  # [T, B, heads, N, head_dim]
        # ============================================================
        
        # Reshape back: [T, B, N, D]
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)
        
        # Output projection
        out_flat = out.reshape(T * B * N, D)
        out = self.proj_linear(out_flat)
        out = self.proj_bn(out).reshape(T, B, N, D)
        
        # Residual connection
        out = out + identity
        
        return out


# ============================================================================
# SWS Block with SDSA (always enabled)
# ============================================================================

class SWS_SDSA(nn.Module):
    """
    Spiking Wavelet Selector with Spike-Driven Self-Attention.
    
    Architecture:
    1. Wavelet Branch: Haar transform → SpikF selector → Inverse Haar
    2. SDSA Branch: Spike-Driven Self-Attention over patches
    3. Conv Branches: Intra-conv + Inter-conv
    """
    def __init__(
        self,
        patch_num: int,
        D: int,
        patch_dim: int,
        tau: float,
        alpha: float,
        num_heads: int = 8,
        keep_levels: KeepLevels = "all",
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        if not _is_power_of_two(patch_num):
            raise ValueError(f"patch_num must be power of 2 for Haar, got {patch_num}")

        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.D = D

        # Haar transform modules
        self.haar_fwd = HaarForward(n=patch_num, spike=transform_spike, vth=transform_vth)
        self.haar_inv = HaarInverse(n=patch_num, spike=transform_spike, vth=transform_vth)

        # Level mask
        level_mask_1d = make_haar_level_mask(patch_num, keep_levels)
        self.register_buffer("level_mask", level_mask_1d.view(1, 1, 1, 1, patch_num), persistent=False)

        # SpikF-style selector
        self.time2wave = nn.Linear(patch_num, patch_num)

        # Spike-Driven Self-Attention (always enabled)
        self.sdsa = SDSA(dim=patch_dim, num_heads=num_heads, tau=tau)

        # Convolutions
        self.intra_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[5, 1], stride=[1, 1], padding=[2, 0])
        self.inter_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[3, 1], stride=[1, 1], padding=[1, 0])

        # Spiking neurons
        self.generator_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch", v_threshold=0.1)
        self.mp_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.sws_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")
        self.inter_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="torch")

        # Batch norms
        self.bn1 = nn.BatchNorm2d(patch_dim)
        self.bn2 = nn.BatchNorm2d(patch_dim)
        self.bn3 = nn.BatchNorm2d(patch_dim)
        self.bn4 = nn.BatchNorm2d(patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, pd, pn, D]
        returns: [T, B, pd, pn, D]
        """
        res_x = x
        T, B, pd, pn, D = x.shape

        # ========== Wavelet Branch ==========
        x_t = x.transpose(-1, -2).contiguous()  # [T, B, pd, D, pn]
        wav = self.haar_fwd(x_t)

        # SpikF-style selector
        selector = self.time2wave(x_t)
        selector = selector.flatten(0, 1)
        selector = self.bn1(selector)
        selector = selector.view(T, B, pd, D, pn)
        selector = self.generator_lif(selector)
        selector = selector.sum(dim=0, keepdim=True)
        selector = self.mp_lif(selector)
        selector = selector.repeat(T, 1, 1, 1, 1).float()

        # Apply level mask
        selector = selector * self.level_mask
        wav = wav * self.level_mask

        # Select and inverse
        remain = selector * wav
        current = self.haar_inv(remain)

        current = current.transpose(-1, -2).contiguous()
        current = current.flatten(0, 1)
        current = self.bn2(current)
        current = current.view(T, B, pd, pn, D)

        spike = self.sws_lif(current)
        x = spike + res_x
        res_x = x

        # ========== SDSA Branch ==========
        # Reshape: [T, B, pd, pn, D] -> apply SDSA over patches for each feature dim
        # Process each feature dimension D separately
        x_sdsa = x.permute(0, 1, 4, 3, 2).contiguous()  # [T, B, D, pn, pd]
        T, B, D_feat, pn, pd = x_sdsa.shape
        
        # Reshape to [T, B*D, pn, pd] for SDSA (pn = tokens, pd = features)
        x_sdsa = x_sdsa.reshape(T, B * D_feat, pn, pd)
        x_sdsa = self.sdsa(x_sdsa)  # [T, B*D, pn, pd]
        
        # Reshape back: [T, B, D, pn, pd] -> [T, B, pd, pn, D]
        x_sdsa = x_sdsa.reshape(T, B, D_feat, pn, pd)
        x_sdsa = x_sdsa.permute(0, 1, 4, 3, 2).contiguous()  # [T, B, pd, pn, D]
        
        x = x_sdsa + res_x
        res_x = x

        # ========== Conv Branches ==========
        # Intra conv
        y = x.flatten(0, 1)
        y = self.intra_conv(y)
        y = self.bn3(y)
        y = y.view(T, B, pd, pn, D)
        y = self.intra_lif(y) + res_x
        res_x = y

        # Inter conv
        z = y.transpose(0, 3).contiguous()
        z = z.flatten(0, 1)
        z = self.inter_conv(z)
        z = self.bn4(z)
        z = z.view(pn, B, pd, T, D).transpose(0, 3).contiguous()
        z = self.inter_lif(z)
        z = z + res_x

        return z


# ============================================================================
# SPE (Spiking Patch Embedding)
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
# Main Model: SwaveNetV1_Attention
# ============================================================================

class SwaveNetV1_Attention(nn.Module):
    """
    Spiking Wavelet Network V1 with Spike-Driven Self-Attention (SDSA).
    
    This is SwaveNetV1 + SDSA integrated. SDSA is always enabled.
    
    Architecture:
    - Haar wavelet transform with ternary spiking
    - SpikF-style frequency selector
    - Spike-Driven Self-Attention (SDSA) for patch attention (87.2× more efficient than vanilla attention)
    - Intra/Inter convolutions
    """
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
        num_heads: int = 8,
        keep_levels: KeepLevels = "all",
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        
        self.T = T
        self.pred_len = pred_len
        
        print(f"[SwaveNetV1_Attention] patch_num={patch_num}, patch_dim={patch_dim}, num_heads={num_heads}, SDSA=enabled")
        
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)

        self.SWSs = nn.ModuleList([
            SWS_SDSA(
                patch_num=patch_num,
                D=D,
                patch_dim=patch_dim,
                tau=tau,
                alpha=alpha,
                num_heads=num_heads,
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

        x = self.SPE(x)
        T, B, pd, pn, D = x.shape

        for blk in self.SWSs:
            x = blk(x)

        # Decode
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(-2, -1)
        x = self.dense1(x)
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(T, B, self.pred_len, D)

        # De-normalize
        x = x * std
        x = x + mean.repeat(T, 1, 1, 1)
        return x