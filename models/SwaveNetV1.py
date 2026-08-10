# SpikF-style Spiking Wavelet Selector (SWS) using Haar (db1) with ternary spiking
import math
from abc import abstractmethod
from typing import Callable, Iterable, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn

from spikingjelly.clock_driven import surrogate, base
from spikingjelly.clock_driven.neuron import MultiStepLIFNode



# adapted from Spiking Wavelet Transformer

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
        return pos_spike - neg_spike  # ∈ {-1, 0, +1} (surrogate during backprop)

    def neuronal_reset(self, spike: torch.Tensor):
        spike_d = spike.detach() if self.detach_reset else spike

        pos = spike_d > 0
        neg = spike_d < 0
        spike_d_vth = pos * self.v_threshold + neg * self.neg_v_threshold

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d_vth
        else:
            # hard reset
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



# 2) Haar matrix + level indexing / masks with ChatGPT's help


def _is_power_of_two(n: int) -> bool:
    return (n > 0) and (n & (n - 1) == 0)

def haar_1d_matrix_np(n: int) -> np.ndarray:
    """Unnormalized n×n Haar (db1) matrix. n must be power of 2."""
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
    """Orthonormal Haar matrix H (rows normalized)."""
    H = torch.from_numpy(haar_1d_matrix_np(n)).to(device=device, dtype=torch.float32)
    norms = torch.linalg.norm(H, dim=1, keepdim=True)
    H = H / (norms + 1e-12)
    return H

def haar_level_slices(n: int) -> List[slice]:
    """
    Coefficient index grouping for Haar matrix transform.

    For n = 2^J:
      - approx (scaling): index 0
      - detail level 1 (coarsest): indices [1 : 2)
      - detail level 2: indices [2 : 4)
      - ...
      - detail level J (finest): indices [2^(J-1) : 2^J)
    Returns a list of slices:
      slices[0] = approx slice
      slices[l] = detail level l slice (for l=1..J)
    """
    if not _is_power_of_two(n):
        raise ValueError(f"n must be power of 2, got {n}")
    J = int(math.log2(n)) # look how J is calculated
    out = [slice(0, 1)]
    for l in range(1, J + 1):
        out.append(slice(2 ** (l - 1), 2 ** l))
    return out

KeepLevels = Union[str, Iterable[int]]


def make_haar_level_mask(n: int, keep_levels: KeepLevels) -> torch.Tensor:
    """
    Returns mask of shape [n] with 1s at kept coefficient indices, 0 elsewhere.

    keep_levels options:
      - "all": keep all coeffs
      - "detail": keep all detail levels (drop approx index 0)
      - "approx": keep only approx (index 0)
      - "finest": keep only finest detail level (level J)
      - iterable of ints: keeps specified levels:
            0 means approx,
            1..J means detail level number
        Example: keep_levels=[0,1] keeps approx + coarsest detail
                 keep_levels=[3] keeps detail level 3 only
    """
    slices = haar_level_slices(n)
    J = len(slices) - 1
    mask = torch.zeros(n, dtype=torch.float32)

    if isinstance(keep_levels, str):
        mode = keep_levels.lower()
        if mode == "all":
            mask[:] = 1.0
        elif mode == "detail":
            mask[slices[1].start : slices[J].stop] = 1.0  # indices 1..n-1
        elif mode == "approx":
            mask[slices[0]] = 1.0
        elif mode == "finest":
            mask[slices[J]] = 1.0
        else:
            raise ValueError(f"Unknown keep_levels='{keep_levels}'. Use all/detail/approx/finest or a list of ints.")
    else:
        for lvl in keep_levels:
            if not isinstance(lvl, int):
                raise TypeError("keep_levels iterable must contain ints (0..J)")
            if lvl < 0 or lvl > J:
                raise ValueError(f"level {lvl} out of range [0..{J}] for n={n}")
            mask[slices[lvl]] = 1.0

    return mask

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


# 3) Haar forward/inverse modules


class HaarForward(nn.Module):
    """
    Forward Haar along last dimension: y = x @ H.T
    Optionally applies ternary spiking after transform.
    """
    def __init__(
        self,
        n: int,
        spike: bool = True,
        vth: float = 1.0,
        detach_reset: bool = False,
    ):
        super().__init__()
        self.n = n
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth),
            neg_v_threshold=float(vth),
            v_reset=0.0,
            detach_reset=detach_reset,
        ) if spike else nn.Identity()

        self.register_buffer("H", torch.empty(0), persistent=False)
        self._built = False

    def _build(self, device: torch.device):
        H = haar_matrix(self.n, device)
        self.H = H
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, ..., n] (Haar on last dim)
        if not self._built or self.H.numel() == 0 or self.H.device != x.device:
            self._build(x.device)
        y = torch.matmul(x, self.H.T)  # [T, ..., n]
        return self.neuron(y)


class HaarInverse(nn.Module):
    """
    Inverse Haar along last dimension: x = y @ H
    Optionally applies ternary spiking after inverse.
    """
    def __init__(
        self,
        n: int,
        spike: bool = True,
        vth: float = 1.0,
        detach_reset: bool = False,
    ):
        super().__init__()
        self.n = n
        self.spike = spike
        self.neuron = MultiStepNegIFNode(
            v_threshold=float(vth),
            neg_v_threshold=float(vth),
            v_reset=0.0,
            detach_reset=detach_reset,
        ) if spike else nn.Identity()

        self.register_buffer("H", torch.empty(0), persistent=False)
        self._built = False

    def _build(self, device: torch.device):
        H = haar_matrix(self.n, device)
        self.H = H
        self._built = True

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [T, ..., n]
        if not self._built or self.H.numel() == 0 or self.H.device != y.device:
            self._build(y.device)
        x = torch.matmul(y, self.H)   # [T, ..., n]
        return self.neuron(x)



# SpikF-style SWS block

class SWS(nn.Module):
    """
    SpikF-style Spiking Wavelet Selector:
    """
    def __init__(
        self,
        patch_num: int,
        D: int,
        patch_dim: int,
        tau: float,
        alpha: float, 
        keep_levels: KeepLevels = "all",
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        if not _is_power_of_two(patch_num):
            raise ValueError(f"patch_num must be power of 2 for Haar, got {patch_num}")

        self.patch_num = patch_num

        # Haar transform modules on pn dimension (last dim after transpose in forward)
        self.haar_fwd = HaarForward(
            n=patch_num,
            spike=transform_spike,
            vth=transform_vth,
            detach_reset=False,
        )
        self.haar_inv = HaarInverse(
            n=patch_num,
            spike=transform_spike,
            vth=transform_vth,
            detach_reset=False,
        )

        # Fixed level mask (shape [1,1,1,1,pn]) for broadcasting onto [T,B,pd,D,pn]
        level_mask_1d = make_haar_level_mask(patch_num, keep_levels)  # [pn]
        self.register_buffer("level_mask", level_mask_1d.view(1, 1, 1, 1, patch_num), persistent=False)

        # SpikF-style selector mapping pn->pn (instead of pn->pn//2+1 for rfft)
        self.time2wave = nn.Linear(patch_num, patch_num)

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

        # Batch norms (channels = patch_dim = pd)
        self.bn1 = nn.BatchNorm2d(patch_dim)  # selector BN
        self.bn2 = nn.BatchNorm2d(patch_dim)  # time-domain BN after inverse
        self.bn3 = nn.BatchNorm2d(patch_dim)  # intra conv BN
        self.bn4 = nn.BatchNorm2d(patch_dim)  # inter conv BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, pd, pn, D]
        returns: [T, B, pd, pn, D]
        """
        res_x = x
        T, B, pd, pn, D = x.shape
        assert pn == self.patch_num, f"Expected pn={self.patch_num}, got {pn}"

        #  [T,B,pd,D,pn]
        x_t = x.transpose(-1, -2).contiguous()

        
        wav = self.haar_fwd(x_t)  # [T,B,pd,D,pn]

        # SpikF-style selector 
        selector = self.time2wave(x_t)          # [T,B,pd,D,pn]
        selector = selector.flatten(0, 1)       # [T*B,pd,D,pn]
        selector = self.bn1(selector)
        selector = selector.view(T, B, pd, D, pn)
        selector = self.generator_lif(selector)
        selector = selector.sum(dim=0, keepdim=True)     # [1,B,pd,D,pn]
        selector = self.mp_lif(selector)
        selector = selector.repeat(T, 1, 1, 1, 1).float()  # [T,B,pd,D,pn]

        # Apply Haar level mask 
        selector = selector * self.level_mask
        wav = wav * self.level_mask

        # Select wavelet coefficients
        remain = selector * wav  # [T,B,pd,D,pn]

        # Inverse Haar 
        current = self.haar_inv(remain)

        # [T,B,pd,pn,D]
        current = current.transpose(-1, -2).contiguous()
        current = current.flatten(0, 1)        # [T*B,pd,pn,D]
        current = self.bn2(current)
        current = current.view(T, B, pd, pn, D)


        spike = self.sws_lif(current)
        x = spike + res_x
        res_x = x

        # Intra conv branch
        y = x.flatten(0, 1)        # [T*B,pd,pn,D]
        y = self.intra_conv(y)
        y = self.bn3(y)
        y = y.view(T, B, pd, pn, D)
        y = self.intra_lif(y) + res_x
        res_x = y

        # Inter conv branch 
        z = y.transpose(0, 3).contiguous()     # [pn,B,pd,T,D]
        z = z.flatten(0, 1)                    # [pn*B,pd,T,D]
        z = self.inter_conv(z)
        z = self.bn4(z)
        z = z.view(pn, B, pd, T, D).transpose(0, 3).contiguous()  # [T,B,pd,pn,D]
        z = self.inter_lif(z)
        z = z + res_x

        return z


# from SPikF

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


# Main model

class SwaveNetV1(nn.Module):
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
        keep_levels: KeepLevels = [0,1],      # "all" | "detail" | "approx" | "finest" | list of ints
        transform_spike: bool = True,
        transform_vth: float = 1.0,
    ):
        super().__init__()
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)

        self.SWSs = nn.ModuleList([
            SWS(
                patch_num=patch_num,
                D=D,
                patch_dim=patch_dim,
                tau=tau,
                alpha=alpha,
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

        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        returns: [T, B, pred_len, D]  (same as SpikF)
        """
       
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        x = self.SPE(x)  # [T, B, pd, pn, D]
        T, B, pd, pn, D = x.shape

        for blk in self.SWSs:
            x = blk(x)

        # Decode (same as SpikF)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # [T,B,D,pd,pn]
        x = x.flatten(-2, -1)                      # [T,B,D,pd*pn]
        x = self.dense1(x)                         # [T,B,D,hidden]
        x = x.flatten(0, 1)                        # [T*B,D,hidden]
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)                         # [T*B,D,pred_len]
        x = x.transpose(-1, -2).contiguous()        # [T*B,pred_len,D]
        x = x.view(T, B, self.pred_len, D)

        # De-normalize
        x = x * std
        x = x + mean.repeat(T, 1, 1, 1)
        return x


