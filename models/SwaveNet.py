from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import numpy as np

from spikingjelly.clock_driven import surrogate, base
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# TERNARY SPIKING NEURONS (from SWformer)

class NegBaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('neg_v_threshold', neg_v_threshold)
        self.register_memory('v_reset', v_reset)
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.register_memory('inhibit', 1e-3)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        pos_spike = self.surrogate_function(self.v - self.v_threshold)
        neg_spike = self.surrogate_function(-(self.v + self.neg_v_threshold + self.inhibit))
        return pos_spike - neg_spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        pos_spike = spike_d > 0
        neg_spike = spike_d < 0
        spike_d_vth = pos_spike * self.v_threshold + neg_spike * self.neg_v_threshold

        if self.v_reset is None:
            self.v = self.v - spike_d_vth
        else:
            self.v = (1. - spike_d.abs()) * self.v + spike_d * self.v_reset

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike


class NegIFNode(NegBaseNode):
    """Integrate-and-Fire neuron with ternary output."""
    
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class MultiStepNegIFNode(NegIFNode):
    """Multi-step ternary neuron for temporal sequences."""
    
    def __init__(self, v_threshold: float = 0.5, neg_v_threshold: float = 0.5, v_reset: float = 0., 
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 backend='torch'):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq

    def reset(self):
        super().reset()
        self.v_seq = None

# HAAR WAVELET TRANSFORMS

def haar_1d_matrix(n):
    """Generate n x n Haar matrix. n must be power of 2."""
    if n == 1:
        return np.array([[1.0]])
    
    H_next = haar_1d_matrix(n // 2)
    upper = np.kron(H_next, [1, 1])
    lower = np.kron(np.eye(len(H_next)), [1, -1])
    return np.vstack((upper, lower))


def haar_matrix(N, device):
    """Generate normalized orthogonal Haar matrix."""
    H = torch.tensor(haar_1d_matrix(N), dtype=torch.float32)
    norms = torch.linalg.norm(H, axis=1, keepdims=True)
    H = H / norms
    return H.to(device)


class Haar1DForwardTS(nn.Module):
    """1D Haar forward transform for time-series."""
    
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_neuron = neuron_type(v_threshold=vth)
        self.H = None

    def build(self, N, device):
        self.H = haar_matrix(N, device)

    def forward(self, x):
        if self.H is None or self.H.shape[0] != x.shape[-2]:
            self.build(x.shape[-2], x.device)
        if self.H.device != x.device:
            self.H = self.H.to(x.device)
        haar = torch.matmul(self.H, x)
        return self.haar_neuron(haar)


class Haar1DInverseTS(nn.Module):
    """1D Haar inverse transform for time-series."""
    
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_inv_neuron = neuron_type(v_threshold=vth)
        self.H = None

    def build(self, N, device):
        self.H = haar_matrix(N, device)

    def forward(self, x):
        if self.H is None or self.H.shape[0] != x.shape[-2]:
            self.build(x.shape[-2], x.device)
        if self.H.device != x.device:
            self.H = self.H.to(x.device)
        haar_inv = torch.matmul(self.H.T, x)
        return self.haar_inv_neuron(haar_inv)


# SPIKING WAVELET SELECTOR (SWS) 


class Erode(nn.Module):
    """Max pooling for spike emphasis."""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (T, B, pd, pn, D)
        shape = x.shape
        x = x.permute(0, 1, 2, 4, 3).contiguous()  # (T, B, pd, D, pn)
        x = x.flatten(0, 3).unsqueeze(1)  # (T*B*pd*D, 1, pn)
        x = self.pool(x).squeeze(1)
        x = x.reshape(shape[0], shape[1], shape[2], shape[4], shape[3])
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        return x


class SWS(nn.Module):
    def __init__(self, patch_num, D, patch_dim, tau, alpha, num_blocks=16, scale_select='both'):
        super().__init__()
        
        assert scale_select in ['both', 'detail'], \
            f"scale_select must be 'both' or 'detail', got {scale_select}"
        
        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.num_blocks = num_blocks
        self.block_size = patch_dim // num_blocks
        self.scale_select = scale_select  # NEW: scale selection mode
        
        # Wavelet transforms with ternary neurons
        self.haar_forward = Haar1DForwardTS(MultiStepNegIFNode, vth=1.0)
        self.haar_inverse = Haar1DInverseTS(MultiStepNegIFNode, vth=1.0)
        
        # Batch norms
        self.haar_forward_bn = nn.BatchNorm2d(patch_dim)
        self.haar_weight_bn = nn.BatchNorm2d(patch_dim)
        self.haar_inverse_bn = nn.BatchNorm2d(patch_dim)
        
        # Learnable block-diagonal weights
        self.scale = 0.02
        self.haar_weight = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
        )
        
        # LIF neurons
        self.x_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.haar_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.inter_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        
        # Convolutions (same as SpikF)
        self.intra_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[5, 1], padding=[2, 0])
        self.inter_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[3, 1], padding=[1, 0])
        self.bn_intra = nn.BatchNorm2d(patch_dim)
        self.bn_inter = nn.BatchNorm2d(patch_dim)
        
        self.pool = Erode()

    def multiply(self, input, weights):
        """Block-diagonal multiplication."""
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        res_x = x
        T, B, pd, pn, D = x.shape
        
        # Wavelet path
        x = self.x_neuron(x)
        x = self.pool(x)
        
        # Haar forward - produces coefficients
        # First half (0 to pn//2): Approximation coefficients (low-frequency)
        # Second half (pn//2 to pn): Detail coefficients (high-frequency)
        haar = self.haar_forward(x)
        

        if self.scale_select == 'detail':
            # Zero out approximation coefficients, keep only detail
            half = pn // 2
            haar = haar.clone()
            haar[:, :, :, :half, :] = 0
        # else 'both': keep all coefficients 
       
        
        haar_flat = haar.flatten(0, 1)
        haar_flat = self.haar_forward_bn(haar_flat)
        haar = haar_flat.reshape(T, B, pd, pn, D)
        haar = self.haar_neuron(haar)
        
        # Block-diagonal weighting
        # Reshape: (T, B, pd, pn, D) -> (T*B*pn*D, num_blocks, block_size)
        haar = haar.permute(0, 1, 3, 4, 2).contiguous()  # (T, B, pn, D, pd)
        haar = haar.reshape(T * B * pn * D, self.num_blocks, self.block_size)
        haar = self.multiply(haar, self.haar_weight)
        
        # Reshape back: (T*B*pn*D, num_blocks, block_size) -> (T, B, pd, pn, D)
        haar = haar.reshape(T, B, pn, D, self.patch_dim)
        haar = haar.permute(0, 1, 4, 2, 3).contiguous()  # (T, B, pd, pn, D)
        
        haar_flat = haar.flatten(0, 1)
        haar_flat = self.haar_weight_bn(haar_flat)
        haar = haar_flat.reshape(T, B, pd, pn, D)
        
        # Haar inverse
        haar = self.haar_inverse(haar)
        haar_flat = haar.flatten(0, 1)
        haar_flat = self.haar_inverse_bn(haar_flat)
        haar = haar_flat.reshape(T, B, pd, pn, D)
        haar = self.pool(haar)
        
        # Convolution path (same as SpikF)
        x = haar + res_x
        res_x = x
        
        # Intra conv
        x = x.flatten(0, 1)
        x = self.intra_conv(x)
        x = self.bn_intra(x)
        x = x.reshape(T, B, pd, pn, D)
        x = self.intra_lif(x) + res_x
        res_x = x
        
        # Inter conv
        x = x.permute(3, 1, 2, 0, 4).contiguous()  # (pn, B, pd, T, D)
        x = x.flatten(0, 1)  # (pn*B, pd, T, D)
        x = self.inter_conv(x)
        x = self.bn_inter(x)
        x = x.reshape(pn, B, pd, T, D)
        x = x.permute(3, 1, 2, 0, 4).contiguous()  # (T, B, pd, pn, D)
        x = self.inter_lif(x) + res_x
        
        return x

# SPIKING PATCH ENCODER

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
        x = x.reshape(B, self.patch_num, L // self.patch_num, D)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, patch_num, D, patch_size)
        x = self.patch_projector(x)  # (B, patch_num, D, patch_dim)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, B, patch_num, D, patch_dim)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (T, B, patch_dim, patch_num, D)
        
        # Batch norm
        x = x.flatten(0, 1)  # (T*B, patch_dim, patch_num, D)
        x = self.bn(x)
        x = x.reshape(self.T, B, self.patch_dim, self.patch_num, D)
        
        x = self.encoder_lif(x)
        return x

# MAIN MODEL

class SwaveNet(nn.Module):
    def __init__(self, input_len, patch_num, patch_dim, T, levels, D, pred_len, tau, alpha, hidden_dim, scale_select='both'):
        super().__init__()
        
        self.T = T
        self.pred_len = pred_len
        self.D = D
        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.scale_select = scale_select  # NEW
        
        # Encoder (same as SpikF)
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)
        
        # Feature extraction: SWS replaces SFS
        self.SWSs = nn.ModuleList([
            SWS(patch_num, D, patch_dim, tau, alpha, scale_select=scale_select)  # Pass scale_select
            for _ in range(levels)
        ])
        
        # Decoder (same as SpikF)
        self.dense1 = nn.Linear(patch_num * patch_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, pred_len)
        self.bn = nn.BatchNorm1d(D)
        self.activ = nn.GELU()

    def forward(self, x):
        # Instance normalization
        mean = x.mean(dim=1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std
        
        # Encode
        x = self.SPE(x)  # (T, B, pd, pn, D)
        T, B, pd, pn, D = x.shape
        
        # SWS blocks
        for sws in self.SWSs:
            x = sws(x)
        
        # Decode
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (T, B, D, pd, pn)
        x = x.flatten(-2, -1)  # (T, B, D, pd*pn)
        x = self.dense1(x)  # (T, B, D, hidden_dim)
        x = x.flatten(0, 1)  # (T*B, D, hidden_dim)
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)  # (T*B, D, pred_len)
        x = x.permute(0, 2, 1).contiguous()  # (T*B, pred_len, D)
        x = x.reshape(T, B, self.pred_len, D)  # (T, B, pred_len, D)
        
        # Denormalize
        x = x * std
        x = x + mean
        
        return x