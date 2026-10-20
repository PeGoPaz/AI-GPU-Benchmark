"""The card's peak dense matrix throughput, from its architecture and clock.

Estimated TFLOPS alone does not compare across cards — 40 TFLOPS is superb
on one and poor on another. Model FLOPs utilisation (MFU), achieved over
peak, does. The peak is computed rather than looked up by name:

    peak = SMs × max SM clock × FLOPs per clock per SM

The per-SM rate is fixed by the architecture, which compute capability
identifies; SM count and clock come from the card itself, so laptop parts,
factory overclocks and cut-down dies all get their own peak without a table
of product names to keep current. NVIDIA's whitepaper figures fall out of
the same formula, and the tests pin them.

One wrinkle: GeForce cards run 16-bit tensor maths with FP32 accumulate —
what PyTorch's BF16 and FP16 matmuls use — at half the rate of the
workstation and datacenter parts built on the same silicon.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Architecture:
    name: str
    # Dense FLOPs per clock per SM: 16-bit tensor-core matmul with FP32
    # accumulate on the full-rate parts, and plain FP32 on the CUDA cores.
    tensor_16bit: int
    fp32: int
    bf16: bool = True               # BF16 tensor support, Ampere onwards


ARCHITECTURES: dict[tuple[int, int], Architecture] = {
    (7, 0): Architecture("Volta", tensor_16bit=1024, fp32=128, bf16=False),
    (7, 5): Architecture("Turing", tensor_16bit=1024, fp32=128, bf16=False),
    (8, 0): Architecture("Ampere (GA100)", tensor_16bit=2048, fp32=128),
    (8, 6): Architecture("Ampere", tensor_16bit=512, fp32=256),
    (8, 9): Architecture("Ada Lovelace", tensor_16bit=1024, fp32=256),
    (9, 0): Architecture("Hopper", tensor_16bit=4096, fp32=256),
    (12, 0): Architecture("Blackwell", tensor_16bit=1024, fp32=256),
}


def _halved_on_geforce(name: str) -> bool:
    """GeForce parts halve 16-bit FP32-accumulate tensor maths; TITANs never did."""
    return "geforce" in name.lower()


@dataclass(frozen=True)
class Peak:
    tflops: float
    architecture: str
    unit: str                       # "16-bit tensor" or "FP32"


def peak_tflops(device: dict[str, Any] | None, sm_count: int | None,
                compute_dtype: str | None) -> Peak | None:
    """Dense peak for this card in the dtype the run computed in.

    None when anything is unknown — the compute capability, the clock, the
    SM count, an architecture not in the table — or when the card cannot run
    the dtype in hardware (BF16 before Ampere): a made-up peak would make
    MFU meaningless rather than merely approximate.
    """
    if not device or not sm_count or not compute_dtype:
        return None
    capability = device.get("compute_capability")
    clock_mhz = device.get("max_sm_clock_mhz")
    if not capability or not clock_mhz:
        return None
    arch = ARCHITECTURES.get((int(capability[0]), int(capability[1])))
    if arch is None:
        return None

    if compute_dtype == "float32":
        per_sm, unit = arch.fp32, "FP32"
    else:
        if compute_dtype == "bfloat16" and not arch.bf16:
            return None
        per_sm, unit = arch.tensor_16bit, "16-bit tensor"
        if _halved_on_geforce(str(device.get("name", ""))):
            per_sm //= 2
    return Peak(tflops=sm_count * clock_mhz * 1e6 * per_sm / 1e12,
                architecture=arch.name, unit=unit)
