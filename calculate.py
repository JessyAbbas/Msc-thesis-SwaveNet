from math import ceil
import numpy
from numbers import Real



#---------------------------------------------------------------------------------------------------------------------------------------

PJ_PER_MAC = 3.2  # pJ per MAC (mul+acc together)
PJ_PER_ADD = 0.1 

#PJ_PER_MAC = 4.6  # pJ per MAC (mul+acc together)
#PJ_PER_ADD = 0.9

#---------------------------------------------------------------------------------------------------------------------------------------


"""
Input:
 Binary:
            input_events = x.sum()

        Ternary or signed integer:
            input_events = x.abs().sum()

        Nonnegative integer event multiplicity:
            input_events = x.sum()

Output:

        Binary:
            y.sum()

        Ternary:
            y.abs().sum()

"""


# SFFT, S-iFFT, ANN FFT, ANN iFFT cost (µJ)

#---------------------------------------------------------------------------------------------------------------------------------------

def factor_m_2n(L: int):
    # L: length of the input signal
    n = 0
    m = L
    while m % 2 == 0:
        m //= 2
        n += 1
    return m, n



def sfft_sops(L, alpha, T, dim_1, dim_2):
    """
    SFFT cost (µJ)
    L: length of the input signal (patch num in SpikF case)
    alpha: firing rate
    T: number of time steps
    dim_1, dim_2: dimensions of the input signal (patch_dim in SpikF case, channels) - basicallty excluding batch size, other dimension that not covered by previous ones above.
    """
    m, n = factor_m_2n(L)
    beta = 1.0 - alpha
    s = 0.0
    for k in range(1, n + 1):
        mk = m * (2 ** (k - 1))
        term = 8.0 * (1.0 - beta ** mk) ** 2 + 6.0 * (beta ** mk) * (1.0 - beta ** mk)
        s += L * term
    s += L * (2.0 * m * alpha - 2.0 + 2.0 * (beta ** m))
    return s*dim_1*dim_2*T*PJ_PER_ADD*1e-6


def sifft_sops(L, alpha, T, dim_1, dim_2):
    """
    S-iFFT cost (µJ)
    L: length of the input signal (patch num in SpikF case)
    alpha: firing rate
    T: number of time steps
    dim_1, dim_2: dimensions of the input signal (patch_dim in SpikF case, channels) - basicallty excluding batch size, other dimension that not covered by previous ones above.
    """
    m, n = factor_m_2n(L)
    beta = 1.0 - alpha
    s = 0.0
    for k in range(1, n + 1):
        mk = m * (2 ** (k - 1))
        term = 8.0 * (1.0 - beta ** mk) ** 2 + 6.0 * (beta ** mk) * (1.0 - beta ** mk)
        s += L * term
    s += L * (2.0 * m * alpha - 2.0 + 2.0 * (beta ** m))
    s += 6.0 * L * m * alpha
    return s*dim_1*dim_2*T*PJ_PER_ADD*1e-6



def fft(L, dim_1, dim_2):
    """
    ANN FFT cost (µJ)
    L: length of the input signal 
    dim_1, dim_2: dimensions of the input signal (1 in FGN case, channels) - basicallty excluding batch size, other dimension that not covered by previous ones above.
    """
    m, n = factor_m_2n(L)
    return L*(8*m+8*n -2)*dim_1*dim_2*PJ_PER_MAC*1e-6


def ifft(L, dim_1, dim_2):
    """
    ANN iFFT cost (µJ)
    L: length of the input signal 
    dim_1, dim_2: dimensions of the input signal (1 in FGN case, channels) - basicallty excluding batch size, other dimension that not covered by previous ones above.
    """
    m, n = factor_m_2n(L)
    return L*(8*m+8*n -2)*dim_1*dim_2*PJ_PER_MAC*1e-6

#---------------------------------------------------------------------------------------------------------------------------------------




# Operational Cost (µJ)

#---------------------------------------------------------------------------------------------------------------------------------------


# ANN:

def conv2d_ann(C_in, H_out, W_out, C_out, H_kernel, W_kernel, groups=1, bias = True):
    """
    C_in: input channels
    H_out: output height
    W_out: output width
    C_out: output channels
    H_kernel: kernel height
    W_kernel: kernel width
    groups: number of groups in the convolution
    bias: whether the convolution has a bias term
    """
    inputs_per_filter = (C_in // groups) * H_kernel * W_kernel
    macs=(C_out*H_out*W_out*inputs_per_filter)*PJ_PER_MAC*1e-6
    if bias:
        return macs + (C_out*H_out*W_out)*PJ_PER_ADD*1e-6
    return macs

def fnn_ann(N_in, N_out, position_1, position_2, bias = True):
    """
    N_in: input features
    N_out: output features
    position_1, position_2: spatial dimensions (channels, other dimension that not covered by previous ones above)
    bias: whether the linear layer has a bias term
    """
    macs = (N_in*position_1*position_2*N_out)*PJ_PER_MAC*1e-6
    if bias:
        return macs + (N_out*position_1*position_2)*PJ_PER_ADD*1e-6
    return macs





# SNN:

def conv2d_snn(
    T,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    stride_h,
    stride_w,
    input_spikes,
    output_spikes=None,
    groups=1,
    bias=True,
    lif_leak=True,
    count_reset=True,
):
    """
    Operational energy of an event-driven spiking Conv2d, per sample.

    input_spikes:
        Total number of input spikes for one sample,
        accumulated across all T timesteps, input channels,
        height, and width.

    output_spikes:
        Total number of output spikes for one sample,
        accumulated across all T timesteps and output positions.
    """

    if groups <= 0:
        raise ValueError("groups must be positive.")

    if C_out % groups != 0:
        raise ValueError("C_out must be divisible by groups.")

    output_neurons = C_out * H_out * W_out

    # An input channel connects only to output channels in its group.
    connected_output_channels = C_out // groups

    spatial_fanout = (
        ceil(H_kernel / stride_h)
        * ceil(W_kernel / stride_w)
    )

    fanout_per_input_spike = (
        spatial_fanout * connected_output_channels
    )

    # One leakage MAC per output neuron per timestep for LIF.
    mac_count = T * output_neurons if lif_leak else 0

    # Synaptic accumulations caused by incoming spikes.
    acc_count = input_spikes * fanout_per_input_spike

    # Bias integration at every timestep.
    if bias:
        acc_count += T * output_neurons

    # Membrane reset when an output spike occurs.
    if count_reset:
        acc_count += output_spikes

    return (
        mac_count * PJ_PER_MAC
        + acc_count * PJ_PER_ADD
    ) * 1e-6  # µJ



def fnn_snn(
    T,
    N_out,
    position_1,
    position_2,
    input_spikes,
    output_spikes=None,
    bias=True,
    lif_leak=True,
    count_reset=True,
):
    """
    Operational energy of an event-driven spiking linear layer,
    per sample.

    input_spikes:
        Total input-spike count for one sample across T,
        all input features, and all positions.

    output_spikes:
        Total output-spike count for one sample across T,
        all output features, and all positions.
    """

    positions = position_1 * position_2
    output_neurons = N_out * positions

    # LIF leakage/update.
    mac_count = T * output_neurons if lif_leak else 0

    # Every incoming spike fans out to all N_out neurons.
    acc_count = input_spikes * N_out

    if bias:
        acc_count += T * output_neurons

    if count_reset:
        acc_count += output_spikes

    return (
        mac_count * PJ_PER_MAC
        + acc_count * PJ_PER_ADD
    ) * 1e-6  # µJ

#---------------------------------------------------------------------------------------------------------------------------------------








# Adressing cost (µJ)
#---------------------------------------------------------------------------------------------------------------------------------------

# ANN:


def addr_conv2d_ann(
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
):
    """
    Addressing energy of a dense ANN Conv2d, per sample.

    Input-feature-map index
    Output-feature-map index
    Kernel-weight index

    Addressing ACC count:
        C_in * H_in * W_in
        + C_out * H_out * W_out
        + C_out * H_kernel * W_kernel
    """
    values = {
        "C_in": C_in,
        "H_in": H_in,
        "W_in": W_in,
        "C_out": C_out,
        "H_out": H_out,
        "W_out": W_out,
        "H_kernel": H_kernel,
        "W_kernel": W_kernel,
    }

    for name, value in values.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )

    input_index_acc = C_in * H_in * W_in
    output_index_acc = C_out * H_out * W_out
    kernel_index_acc = C_out * H_kernel * W_kernel

    acc_count = (
        input_index_acc
        + output_index_acc
        + kernel_index_acc
    )

    return acc_count * PJ_PER_ADD * 1e-6


def addr_fnn_ann(
    N_in,
    N_out,
    position_1=1,
    position_2=1,
):
    """
    Addressing energy of a dense ANN linear layer, per sample.

    For one linear operation, the paper counts:
        N_in + N_out

    position_1 * position_2 represents the number of
    independent applications of the same linear layer.
    """
    values = {
        "N_in": N_in,
        "N_out": N_out,
        "position_1": position_1,
        "position_2": position_2,
    }

    for name, value in values.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )

    positions = position_1 * position_2

    acc_count = positions * (N_in + N_out)

    return acc_count * PJ_PER_ADD * 1e-6




# SNN:



def addr_conv2d_snn(
    C_out,
    H_kernel,
    W_kernel,
    input_events,
    stride_h=1,
    stride_w=1,
    groups=1,
):
    """
    Addressing energy of an event-driven SNN Conv2d,
    per sample.

    input_events:
        Total event count for one sample, already accumulated
        across all timesteps, input channels and positions.

    For each input event:

    - Two MACs determine the initial 2D output address.
    - ACC index increments enumerate affected output neurons.
    """
    integer_values = {
        "C_out": C_out,
        "H_kernel": H_kernel,
        "W_kernel": W_kernel,
        "stride_h": stride_h,
        "stride_w": stride_w,
        "groups": groups,
    }

    for name, value in integer_values.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )

    if input_events < 0:
        raise ValueError("input_events must be non-negative.")

    if C_out % groups != 0:
        raise ValueError(
            f"C_out={C_out} must be divisible by groups={groups}."
        )

    connected_output_channels = C_out // groups

    spatial_fanout = (
        ceil(H_kernel / stride_h)
        * ceil(W_kernel / stride_w)
    )

    affected_outputs_per_event = (
        connected_output_channels * spatial_fanout
    )

    # Initial output row and column address.
    mac_count = 2 * input_events

    # Increment through all remaining affected destinations.
    acc_count = input_events * affected_outputs_per_event

    return (
        mac_count * PJ_PER_MAC
        + acc_count * PJ_PER_ADD
    ) * 1e-6


def addr_fnn_snn(
    N_out,
    input_events,
):
    """
    Addressing energy of an event-driven SNN linear layer,
    per sample.

    input_events:
        Total event count for one sample, already accumulated
        across T, all input features and all positions.

    """
    if not isinstance(N_out, int) or N_out <= 0:
        raise ValueError(
            f"N_out must be a positive integer; got {N_out!r}."
        )

    if input_events < 0:
        raise ValueError("input_events must be non-negative.")

    acc_count = input_events * N_out

    return acc_count * PJ_PER_ADD * 1e-6
#---------------------------------------------------------------------------------------------------------------------------------------






# Memory Access Cost (µJ)
#---------------------------------------------------------------------------------------------------------------------------------------


BITS_PER_WORD_DEFAULT = 32
FIFO_WORDS_DEFAULT = 1000
PJ_TO_UJ = 1e-6



def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer; got {value!r}."
        )


def _nonnegative(name, value):
    if not isinstance(value, Real) or value < 0:
        raise ValueError(
            f"{name} must be non-negative; got {value!r}."
        )


# SRAM size and access energy

def sram_kb_from_num_words(
    num_words: float,
    bits_per_word: int = BITS_PER_WORD_DEFAULT,
) -> float:
    """
    Convert number of stored words into SRAM capacity in KiB.

    The paper assumes 32-bit words for all stored values,
    including spike messages.
    """
    _nonnegative("num_words", num_words)
    _positive_int("bits_per_word", bits_per_word)

    bytes_total = num_words * bits_per_word / 8.0
    return bytes_total / 1024.0


def eram_from_sram_kb(sram_kb: float) -> float:
    """
    SRAM-access energy in pJ/access.

    Piecewise-linear interpolation through:
        8 KiB    -> 10 pJ
        32 KiB   -> 20 pJ
        1024 KiB -> 100 pJ

    Below 8 KiB:
        clamp to 10 pJ/access.

    Above 1 MiB:
        clamp to 100 pJ/access, interpreted as banking the
        memory into arrays no larger than approximately 1 MiB.
    """
    _nonnegative("sram_kb", sram_kb)

    if sram_kb <= 8.0:
        return 10.0

    if sram_kb <= 32.0:
        return (
            10.0
            + (sram_kb - 8.0)
            * (20.0 - 10.0)
            / (32.0 - 8.0)
        )

    if sram_kb <= 1024.0:
        return (
            20.0
            + (sram_kb - 32.0)
            * (100.0 - 20.0)
            / (1024.0 - 32.0)
        )

    return 100.0



def memory_energy_from_banks(
    accesses: dict,
    bank_words: dict,
    bits_per_word: int = BITS_PER_WORD_DEFAULT,
):
    """
    Calculate energy separately for every logical SRAM bank.

    accesses format:
        {
            "weights": {"reads": ..., "writes": ...},
            "bias": {"reads": ..., "writes": ...},
            ...
        }

    bank_words format:
        {
            "weights": number_of_stored_words,
            "bias": number_of_stored_words,
            ...
        }

    Returns per-bank details and total energy in µJ.
    """
    details = {}
    total_energy_uj = 0.0
    total_accesses = 0.0

    all_banks = set(accesses) | set(bank_words)

    for bank in all_banks:
        reads = accesses.get(bank, {}).get("reads", 0)
        writes = accesses.get(bank, {}).get("writes", 0)
        words = bank_words.get(bank, 0)

        _nonnegative(f"{bank}.reads", reads)
        _nonnegative(f"{bank}.writes", writes)
        _nonnegative(f"{bank}.words", words)

        access_count = reads + writes

        if words == 0:
            if access_count != 0:
                raise ValueError(
                    f"Bank {bank!r} has accesses but zero capacity."
                )
            continue

        size_kb = sram_kb_from_num_words(
            words,
            bits_per_word,
        )

        energy_per_access_pj = eram_from_sram_kb(size_kb)

        energy_uj = (
            access_count
            * energy_per_access_pj
            * PJ_TO_UJ
        )

        details[bank] = {
            "words": words,
            "size_kb": size_kb,
            "reads": reads,
            "writes": writes,
            "accesses": access_count,
            "energy_per_access_pj": energy_per_access_pj,
            "energy_uj": energy_uj,
        }

        total_accesses += access_count
        total_energy_uj += energy_uj

    return {
        "banks": details,
        "total_accesses": total_accesses,
        "total_energy_uj": total_energy_uj,
    }



def memory_banks_conv2d_ann(
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    groups=1,
    bias=True,
):
    """
    Logical ANN Conv2d SRAM capacities in words, per layer.

    """
    for name, value in {
        "C_in": C_in,
        "H_in": H_in,
        "W_in": W_in,
        "C_out": C_out,
        "H_out": H_out,
        "W_out": W_out,
        "H_kernel": H_kernel,
        "W_kernel": W_kernel,
        "groups": groups,
    }.items():
        _positive_int(name, value)

    if C_in % groups != 0:
        raise ValueError("C_in must be divisible by groups.")

    if C_out % groups != 0:
        raise ValueError("C_out must be divisible by groups.")

    weights = (
        C_out
        * (C_in // groups)
        * H_kernel
        * W_kernel
    )

    return {
        "input": C_in * H_in * W_in,
        "weights": weights,
        "bias": C_out if bias else 0,
        "output": C_out * H_out * W_out,
    }


def mem_counts_conv2d_ann(
    C_in,
    H_out,
    W_out,
    C_out,
    H_kernel,
    W_kernel,
    groups=1,
    bias=True,
):
    """
    ANN Conv2d memory-access counts for one sample.

    The paper assumes no weight/input caching.
    """
    if C_in % groups != 0 or C_out % groups != 0:
        raise ValueError(
            "C_in and C_out must be divisible by groups."
        )

    output_elements = C_out * H_out * W_out

    inputs_per_output = (
        (C_in // groups)
        * H_kernel
        * W_kernel
    )

    synaptic_accesses = (
        output_elements * inputs_per_output
    )

    return {
        "input": {
            "reads": synaptic_accesses,
            "writes": 0,
        },
        "weights": {
            "reads": synaptic_accesses,
            "writes": 0,
        },
        "bias": {
            "reads": output_elements if bias else 0,
            "writes": 0,
        },
        "output": {
            "reads": 0,
            "writes": output_elements,
        },
    }


def mem_conv2d_ann(
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    groups=1,
    bias=True,
    bits_per_word=BITS_PER_WORD_DEFAULT,
):
    banks = memory_banks_conv2d_ann(
        C_in=C_in,
        H_in=H_in,
        W_in=W_in,
        C_out=C_out,
        H_out=H_out,
        W_out=W_out,
        H_kernel=H_kernel,
        W_kernel=W_kernel,
        groups=groups,
        bias=bias,
    )

    counts = mem_counts_conv2d_ann(
        C_in=C_in,
        H_out=H_out,
        W_out=W_out,
        C_out=C_out,
        H_kernel=H_kernel,
        W_kernel=W_kernel,
        groups=groups,
        bias=bias,
    )

    return memory_energy_from_banks(
        accesses=counts,
        bank_words=banks,
        bits_per_word=bits_per_word,
    )





def memory_banks_fnn_ann(
    N_in,
    N_out,
    position_1=1,
    position_2=1,
    bias=True,
):
    """
    Logical ANN Linear SRAM capacities in words.

    The weights are stored once and reused across positions.
    Input and output activation buffers contain every position.
    """
    P = position_1 * position_2

    return {
        "input": N_in * P,
        "weights": N_in * N_out,
        "bias": N_out if bias else 0,
        "output": N_out * P,
    }


def mem_counts_fnn_ann(
    N_in,
    N_out,
    position_1=1,
    position_2=1,
    bias=True,
):
    """
    ANN repeated-Linear memory accesses for one sample.
    """
    P = position_1 * position_2

    return {
        "input": {
            "reads": N_in * P,
            "writes": 0,
        },
        "weights": {
            "reads": N_in * N_out * P,
            "writes": 0,
        },
        "bias": {
            "reads": N_out * P if bias else 0,
            "writes": 0,
        },
        "output": {
            "reads": 0,
            "writes": N_out * P,
        },
    }


def mem_fnn_ann(
    N_in,
    N_out,
    position_1=1,
    position_2=1,
    bias=True,
    bits_per_word=BITS_PER_WORD_DEFAULT,
):
    banks = memory_banks_fnn_ann(
        N_in=N_in,
        N_out=N_out,
        position_1=position_1,
        position_2=position_2,
        bias=bias,
    )

    counts = mem_counts_fnn_ann(
        N_in=N_in,
        N_out=N_out,
        position_1=position_1,
        position_2=position_2,
        bias=bias,
    )

    return memory_energy_from_banks(
        accesses=counts,
        bank_words=banks,
        bits_per_word=bits_per_word,
    )







def memory_banks_conv2d_snn(
    C_in,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    groups=1,
    bias=True,
    stateful_neuron=True,
    fifo_words=FIFO_WORDS_DEFAULT,
):
    """
    Logical SNN Conv2d SRAM capacities in words.

    Potentials are stored once per output neuron. They are
    not multiplied by T because they persist between timesteps.
    """
    if C_in % groups != 0 or C_out % groups != 0:
        raise ValueError(
            "C_in and C_out must be divisible by groups."
        )

    weights = (
        C_out
        * (C_in // groups)
        * H_kernel
        * W_kernel
    )

    output_neurons = C_out * H_out * W_out

    return {
        "fifo": fifo_words,
        "weights": weights,
        "bias": C_out if bias else 0,
        "potentials": (
            output_neurons if stateful_neuron else 0
        ),
    }


def mem_counts_conv2d_snn(
    T,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    stride_h,
    stride_w,
    input_events,
    output_events,
    groups=1,
    bias=True,
    lif_leak=True,
    stateful_neuron=True,
):
    """
    SNN Conv2d memory-access counts for one sample.

    input_events:
        Total input-event multiplicity across all T timesteps.

    output_events:
        Actual output-neuron firing events across all T.
    """
    _nonnegative("input_events", input_events)
    _nonnegative("output_events", output_events)

    if C_out % groups != 0:
        raise ValueError(
            "C_out must be divisible by groups."
        )

    output_neurons = C_out * H_out * W_out

    connected_output_channels = C_out // groups

    spatial_fanout = (
        ceil(H_kernel / stride_h)
        * ceil(W_kernel / stride_w)
    )

    fanout_per_event = (
        connected_output_channels
        * spatial_fanout
    )

    event_updates = input_events * fanout_per_event

    # Bias parameter reads occur once per output neuron
    # at every timestep.
    bias_reads = (
        T * output_neurons
        if bias else 0
    )

    # A full potential traversal is necessary when bias or
    # LIF decay updates every output neuron each timestep.
    dense_potential_updates = (
        T * output_neurons
        if stateful_neuron and (bias or lif_leak)
        else 0
    )

    potential_accesses = (
        event_updates + dense_potential_updates
        if stateful_neuron else 0
    )

    return {
        "fifo": {
            "reads": input_events,
            "writes": output_events,
        },
        "weights": {
            "reads": event_updates,
            "writes": 0,
        },
        "bias": {
            "reads": bias_reads,
            "writes": 0,
        },
        "potentials": {
            "reads": potential_accesses,
            "writes": potential_accesses,
        },
    }


def mem_conv2d_snn(
    T,
    C_in,
    C_out,
    H_out,
    W_out,
    H_kernel,
    W_kernel,
    stride_h,
    stride_w,
    input_events,
    output_events,
    groups=1,
    bias=True,
    lif_leak=True,
    stateful_neuron=True,
    fifo_words=FIFO_WORDS_DEFAULT,
    bits_per_word=BITS_PER_WORD_DEFAULT,
):
    banks = memory_banks_conv2d_snn(
        C_in=C_in,
        C_out=C_out,
        H_out=H_out,
        W_out=W_out,
        H_kernel=H_kernel,
        W_kernel=W_kernel,
        groups=groups,
        bias=bias,
        stateful_neuron=stateful_neuron,
        fifo_words=fifo_words,
    )

    counts = mem_counts_conv2d_snn(
        T=T,
        C_out=C_out,
        H_out=H_out,
        W_out=W_out,
        H_kernel=H_kernel,
        W_kernel=W_kernel,
        stride_h=stride_h,
        stride_w=stride_w,
        input_events=input_events,
        output_events=output_events,
        groups=groups,
        bias=bias,
        lif_leak=lif_leak,
        stateful_neuron=stateful_neuron,
    )

    return memory_energy_from_banks(
        accesses=counts,
        bank_words=banks,
        bits_per_word=bits_per_word,
    )





def memory_banks_fnn_snn(
    N_in,
    N_out,
    position_1=1,
    position_2=1,
    bias=True,
    stateful_neuron=True,
    fifo_words=FIFO_WORDS_DEFAULT,
):
    """
    Logical SNN repeated-Linear SRAM capacities in words.

    Weights are stored once.
    Potentials exist for every output feature and position.
    """
    P = position_1 * position_2

    return {
        "fifo": fifo_words,
        "weights": N_in * N_out,
        "bias": N_out if bias else 0,
        "potentials": (
            N_out * P if stateful_neuron else 0
        ),
    }


def mem_counts_fnn_snn(
    T,
    N_out,
    position_1,
    position_2,
    input_events,
    output_events,
    bias=True,
    lif_leak=True,
    stateful_neuron=True,
):
    """
    SNN repeated-Linear memory-access counts per sample.

    input_events already includes:
        T
        every input feature
        every repeated position
        integer event multiplicity
    """
    _nonnegative("input_events", input_events)
    _nonnegative("output_events", output_events)

    P = position_1 * position_2
    output_neurons = N_out * P

    event_updates = input_events * N_out

    bias_reads = (
        T * output_neurons
        if bias else 0
    )

    dense_potential_updates = (
        T * output_neurons
        if stateful_neuron and (bias or lif_leak)
        else 0
    )

    potential_accesses = (
        event_updates + dense_potential_updates
        if stateful_neuron else 0
    )

    return {
        "fifo": {
            "reads": input_events,
            "writes": output_events,
        },
        "weights": {
            "reads": event_updates,
            "writes": 0,
        },
        "bias": {
            "reads": bias_reads,
            "writes": 0,
        },
        "potentials": {
            "reads": potential_accesses,
            "writes": potential_accesses,
        },
    }


def mem_fnn_snn(
    T,
    N_in,
    N_out,
    position_1,
    position_2,
    input_events,
    output_events,
    bias=True,
    lif_leak=True,
    stateful_neuron=True,
    fifo_words=FIFO_WORDS_DEFAULT,
    bits_per_word=BITS_PER_WORD_DEFAULT,
):
    banks = memory_banks_fnn_snn(
        N_in=N_in,
        N_out=N_out,
        position_1=position_1,
        position_2=position_2,
        bias=bias,
        stateful_neuron=stateful_neuron,
        fifo_words=fifo_words,
    )

    counts = mem_counts_fnn_snn(
        T=T,
        N_out=N_out,
        position_1=position_1,
        position_2=position_2,
        input_events=input_events,
        output_events=output_events,
        bias=bias,
        lif_leak=lif_leak,
        stateful_neuron=stateful_neuron,
    )

    return memory_energy_from_banks(
        accesses=counts,
        bank_words=banks,
        bits_per_word=bits_per_word,
    )
#---------------------------------------------------------------------------------------------------------------------------------------



















































































