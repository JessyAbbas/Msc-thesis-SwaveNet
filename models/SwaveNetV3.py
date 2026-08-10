#SwaveNetV3: Float DWT (No Ternary Neurons)
# Updated: Configurable DWT levels (J) with validation

import torch
import torch.nn as nn
import numpy as np
import math

from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from pytorch_wavelets import DWT1DForward, DWT1DInverse


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_max_dwt_level(signal_length, wavelet='db4'):
    """
    Calculate maximum DWT decomposition level for given signal length.
    
    For DWT, the signal length must be >= filter_length at each level.
    Max level J satisfies: signal_length / 2^J >= filter_length
    
    Args:
        signal_length: Length of input signal
        wavelet: Wavelet name (determines filter length)
    
    Returns:
        max_level: Maximum valid decomposition level
    """
    # Filter lengths for common wavelets
    filter_lengths = {
        'haar': 2, 'db1': 2,
        'db2': 4, 'db3': 6, 'db4': 8, 'db5': 10,
        'db6': 12, 'db7': 14, 'db8': 16,
        'sym2': 4, 'sym3': 6, 'sym4': 8, 'sym5': 10,
        'sym6': 12, 'sym7': 14, 'sym8': 16,
    }
    
    filter_len = filter_lengths.get(wavelet, 8)  # default to 8 if unknown
    
    # Max level: signal_length / 2^J >= filter_length
    # J <= log2(signal_length / filter_length)
    if signal_length <= filter_len:
        return 1
    
    max_level = int(math.floor(math.log2(signal_length / filter_len)))
    return max(1, max_level)


def validate_dwt_level(signal_length, J, wavelet='db4'):
    """
    Validate and potentially adjust DWT level.
    
    Args:
        signal_length: Length of input signal
        J: Requested decomposition level
        wavelet: Wavelet name
    
    Returns:
        valid_J: Valid decomposition level (may be reduced)
        warning: Warning message if level was adjusted, None otherwise
    """
    max_J = get_max_dwt_level(signal_length, wavelet)
    
    if J > max_J:
        warning = f"Requested J={J} exceeds maximum J={max_J} for signal_length={signal_length} with {wavelet}. Using J={max_J}."
        return max_J, warning
    
    return J, None


# ============================================================================
# FLOAT DWT (NO TERNARY NEURONS)
# ============================================================================

class FloatDWT(nn.Module):
    """
    Float DWT using pytorch_wavelets - NO ternary quantization.
    Pure floating point wavelet transform.
    Now supports configurable decomposition levels (J).
    """
    
    def __init__(self, wavelet='haar', mode='symmetric', J=3):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.J = J
        
        # DWT transforms with configurable J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode=mode)
        self.idwt = DWT1DInverse(wave=wavelet, mode=mode)
    
    def forward_transform(self, x):
        """
        Forward DWT (pure float).
        Input: (N, C, L)
        Output: yl (approx), yh (list of detail coefficients at each level)
        """
        yl, yh = self.dwt(x)
        return yl, yh
    
    def inverse_transform(self, yl, yh):
        """
        Inverse DWT (pure float).
        """
        return self.idwt((yl, yh))


# ============================================================================
# SPIKING WAVELET SELECTOR V3 (SWS) - Float DWT + Scale Select
# ============================================================================

class Erode(nn.Module):
    """Max pooling for spike emphasis."""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (T, B, pd, pn, D)
        shape = x.shape
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = x.flatten(0, 3).unsqueeze(1)
        x = self.pool(x).squeeze(1)
        x = x.reshape(shape[0], shape[1], shape[2], shape[4], shape[3])
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        return x


class SWS_V3(nn.Module):
    """
    Spiking Wavelet Selector V3 - Float DWT (no ternary in transform).
    
    Key difference from V1/V2:
    - DWT and iDWT are pure float operations
    - No ternary quantization of wavelet coefficients
    - Still uses LIF neurons for spike processing
    - Supports configurable DWT levels (J) and multi-level detail coefficients
    """
    
    def __init__(self, patch_num, D, patch_dim, tau, alpha, num_blocks=16,
                 scale_select='both', wavelet='haar', dwt_level=3):
        super().__init__()
        
        assert scale_select in ['both', 'detail'], \
            f"scale_select must be 'both' or 'detail', got {scale_select}"
        assert wavelet in ['haar', 'db2', 'db4', 'db8', 'sym4', 'sym8'], \
            f"wavelet must be haar/db2/db4/db8/sym4/sym8, got {wavelet}"
        
        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.num_blocks = num_blocks
        self.block_size = patch_dim // num_blocks
        self.scale_select = scale_select
        self.wavelet = wavelet
        self.dwt_level = dwt_level
        
        # Validate DWT level
        valid_J, warning = validate_dwt_level(patch_num, dwt_level, wavelet)
        if warning:
            print(f"[SWS_V3 Warning] {warning}")
        self.dwt_level = valid_J
        
        # Float DWT with configurable J
        self.float_dwt = FloatDWT(wavelet=wavelet, J=self.dwt_level)
        
        # Batch norms
        self.bn_coeffs = nn.BatchNorm2d(patch_dim)
        self.bn_post = nn.BatchNorm2d(patch_dim)
        
        # Learnable block-diagonal weights
        self.scale = 0.02
        self.haar_weight = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
        )
        
        # LIF neurons 
        self.x_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.coeff_neuron = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.intra_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        self.inter_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='torch')
        
        # Convolutions
        self.intra_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[5, 1], padding=[2, 0])
        self.inter_conv = nn.Conv2d(patch_dim, patch_dim, kernel_size=[3, 1], padding=[1, 0])
        self.bn_intra = nn.BatchNorm2d(patch_dim)
        self.bn_inter = nn.BatchNorm2d(patch_dim)
        
        self.pool = Erode()
        
        # Projection layers (initialized dynamically)
        self.proj_down = None
        self.proj_initialized = False

    def multiply(self, input, weights):
        """Block-diagonal multiplication."""
        return torch.einsum('...bd,bdk->...bk', input, weights)
    
    def _init_projections(self, L_approx, L_details, pn, device):
        """Initialize projection layers based on actual coefficient sizes."""
        if self.proj_initialized:
            return
            
        # Calculate total coefficient length
        total_coeff_len = L_approx + sum(L_details)
        
        # Single projection from all coefficients to patch_num
        self.proj_down = nn.Linear(total_coeff_len, pn).to(device)
        self.proj_initialized = True
        
        print(f"[SWS_V3] Initialized projections: L_approx={L_approx}, L_details={L_details}, total={total_coeff_len} -> pn={pn}")

    def forward(self, x):
        res_x = x
        T, B, pd, pn, D = x.shape
        
        # LIF + pooling (spike processing)
        x_spike = self.x_neuron(x)
        x_spike = self.pool(x_spike)
        
        # Convert spikes to float for DWT
        # Reshape: (T, B, pd, pn, D) -> (T*B*pd*D, 1, pn)
        x_flat = x_spike.permute(0, 1, 4, 2, 3).contiguous()  # (T, B, D, pd, pn)
        x_flat = x_flat.reshape(T * B * D * pd, 1, pn)  # (T*B*D*pd, 1, pn)
        
        # Forward DWT (FLOAT - no ternary)
        yl, yh = self.float_dwt.forward_transform(x_flat)
        # yl: (T*B*D*pd, 1, L_approx)
        # yh: list of J tensors, each (T*B*D*pd, 1, L_detail_j)
        
        L_approx = yl.shape[-1]
        L_details = [yh[j].shape[-1] for j in range(len(yh))]
        
        # Initialize projections if needed
        self._init_projections(L_approx, L_details, pn, x.device)
        
        # Scale selection (BEFORE any processing)
        if self.scale_select == 'detail':
            yl = torch.zeros_like(yl)
        
        # Flatten and concatenate all coefficients
        yl_flat = yl.squeeze(1)  # (T*B*D*pd, L_approx)
        yh_flat = [yh[j].squeeze(1) for j in range(len(yh))]  # list of (T*B*D*pd, L_detail_j)
        
        # Concatenate: [approx, detail_J, detail_J-1, ..., detail_1]
        all_coeffs = [yl_flat] + yh_flat
        coeffs = torch.cat(all_coeffs, dim=-1)  # (T*B*D*pd, total_coeff_len)
        
        # Project to patch_num size
        coeffs = self.proj_down(coeffs)  # (T*B*D*pd, pn)
        
        # Reshape back: (T*B*D*pd, pn) -> (T, B, pd, pn, D)
        coeffs = coeffs.reshape(T, B, D, pd, pn)
        coeffs = coeffs.permute(0, 1, 3, 4, 2).contiguous()  # (T, B, pd, pn, D)
        
        # BatchNorm
        coeffs_flat = coeffs.flatten(0, 1)
        coeffs_flat = self.bn_coeffs(coeffs_flat)
        coeffs = coeffs_flat.reshape(T, B, pd, pn, D)
        
        # Block-diagonal weighting
        coeffs = coeffs.permute(0, 1, 3, 4, 2).contiguous()  # (T, B, pn, D, pd)
        coeffs = coeffs.reshape(T * B * pn * D, self.num_blocks, self.block_size)
        coeffs = self.multiply(coeffs, self.haar_weight)
        coeffs = coeffs.reshape(T, B, pn, D, self.patch_dim)
        coeffs = coeffs.permute(0, 1, 4, 2, 3).contiguous()  # (T, B, pd, pn, D)
        
        # BatchNorm + LIF (convert back to spikes)
        coeffs_flat = coeffs.flatten(0, 1)
        coeffs_flat = self.bn_post(coeffs_flat)
        coeffs = coeffs_flat.reshape(T, B, pd, pn, D)
        coeffs = self.coeff_neuron(coeffs)  # Back to spikes
        
        # Pool
        coeffs = self.pool(coeffs)
        
        # Residual + Conv
        x = coeffs + res_x
        res_x = x
        
        # Intra conv
        x = x.flatten(0, 1)
        x = self.intra_conv(x)
        x = self.bn_intra(x)
        x = x.reshape(T, B, pd, pn, D)
        x = self.intra_lif(x) + res_x
        res_x = x
        
        # Inter conv
        x = x.permute(3, 1, 2, 0, 4).contiguous()
        x = x.flatten(0, 1)
        x = self.inter_conv(x)
        x = self.bn_inter(x)
        x = x.reshape(pn, B, pd, T, D)
        x = x.permute(3, 1, 2, 0, 4).contiguous()
        x = self.inter_lif(x) + res_x
        
        return x


# ============================================================================
# SPIKING PATCH ENCODER 
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
        x = x.reshape(B, self.patch_num, L // self.patch_num, D)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.patch_projector(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.reshape(self.T, B, self.patch_dim, self.patch_num, D)
        
        x = self.encoder_lif(x)
        return x


# ============================================================================
# MAIN MODEL - SwaveNetV3
# ============================================================================

class SwaveNetV3(nn.Module):
    """
    SwaveNet V3: Float DWT (no ternary neurons in transform).
    
    Args:
        input_len: Input sequence length
        patch_num: Number of patches
        patch_dim: Dimension of each patch
        T: Number of timesteps for SNN
        levels: Number of SWS blocks
        D: Number of input dimensions/features
        pred_len: Prediction length
        tau: LIF neuron time constant
        alpha: Not used (kept for compatibility)
        hidden_dim: Hidden dimension for decoder
        scale_select: 'both' (all coeffs) or 'detail' (detail only)
        wavelet: Wavelet type ('haar', 'db2', 'db4', 'db8', 'sym4', 'sym8')
        dwt_level: DWT decomposition level (J)
    """
    
    def __init__(self, input_len, patch_num, patch_dim, T, levels, D, pred_len,
                 tau, alpha, hidden_dim, scale_select='both', wavelet='haar', dwt_level=3):
        super().__init__()
        
        self.T = T
        self.pred_len = pred_len
        self.D = D
        self.patch_num = patch_num
        self.patch_dim = patch_dim
        self.scale_select = scale_select
        self.wavelet = wavelet
        self.dwt_level = dwt_level
        
        # Validate DWT level for the patch_num
        valid_J, warning = validate_dwt_level(patch_num, dwt_level, wavelet)
        if warning:
            print(f"[SwaveNetV3 Warning] {warning}")
        self.dwt_level = valid_J
        
        print(f"[SwaveNetV3] Initialized with wavelet={wavelet}, dwt_level={self.dwt_level}, "
              f"patch_num={patch_num}, levels={levels}")
        
        # Encoder
        self.SPE = SPE(input_len, patch_num, patch_dim, T, tau, D)
        
        # Feature extraction with Float DWT
        self.SWSs = nn.ModuleList([
            SWS_V3(patch_num, D, patch_dim, tau, alpha,
                   scale_select=scale_select, wavelet=wavelet, dwt_level=self.dwt_level)
            for _ in range(levels)
        ])
        
        # Decoder
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
        x = self.SPE(x)
        T, B, pd, pn, D = x.shape
        
        # SWS blocks
        for sws in self.SWSs:
            x = sws(x)
        
        # Decode
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(-2, -1)
        x = self.dense1(x)
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = self.activ(x)
        x = self.dense2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(T, B, self.pred_len, D)
        
        # Denormalize
        x = x * std
        x = x + mean
        
        return x