"""Peak throughput from architecture, SM count and clock — pinned to NVIDIA's own figures."""
import pytest

from gpu_bench.core import metrics
from gpu_bench.core.peak import ARCHITECTURES, peak_tflops


def _card(name, capability, clock_mhz):
    return {"name": name, "compute_capability": list(capability), "max_sm_clock_mhz": clock_mhz}


# (card, SMs, dtype, whitepaper dense TFLOPS): 16-bit tensor with FP32
# accumulate, or plain FP32, at the boost clock the whitepaper quotes.
WHITEPAPER = [
    (_card("NVIDIA GeForce RTX 3090", (8, 6), 1695), 82, "bfloat16", 35.6),
    (_card("NVIDIA GeForce RTX 3090", (8, 6), 1695), 82, "float32", 35.6),
    (_card("NVIDIA RTX A6000", (8, 6), 1800), 84, "float16", 77.4),
    (_card("NVIDIA GeForce RTX 4090", (8, 9), 2520), 128, "bfloat16", 165.2),
    (_card("NVIDIA GeForce RTX 4090", (8, 9), 2520), 128, "float32", 82.6),
    (_card("NVIDIA RTX 6000 Ada Generation", (8, 9), 2505), 142, "bfloat16", 364.2),
    (_card("NVIDIA A100-SXM4-80GB", (8, 0), 1410), 108, "bfloat16", 312.0),
    (_card("NVIDIA A100-SXM4-80GB", (8, 0), 1410), 108, "float32", 19.5),
    (_card("NVIDIA H100 80GB HBM3", (9, 0), 1830), 132, "bfloat16", 989.4),
    (_card("NVIDIA GeForce RTX 5090", (12, 0), 2407), 170, "bfloat16", 209.5),
    (_card("NVIDIA GeForce RTX 2080 Ti", (7, 5), 1545), 68, "float16", 53.8),
    (_card("Tesla V100-SXM2-16GB", (7, 0), 1530), 80, "float16", 125.3),
]


@pytest.mark.parametrize("device, sms, dtype, expected", WHITEPAPER,
                         ids=[f"{d['name']}-{t}" for d, _, t, _ in WHITEPAPER])
def test_peak_matches_the_whitepaper(device, sms, dtype, expected):
    assert peak_tflops(device, sms, dtype).tflops == pytest.approx(expected, rel=0.005)


def test_geforce_halves_the_16_bit_rate_but_not_fp32():
    pro = _card("NVIDIA RTX A6000", (8, 6), 1800)
    geforce = {**pro, "name": "NVIDIA GeForce RTX 3090 Ti"}

    assert peak_tflops(geforce, 84, "bfloat16").tflops == pytest.approx(
        peak_tflops(pro, 84, "bfloat16").tflops / 2)
    assert peak_tflops(geforce, 84, "float32").tflops == peak_tflops(pro, 84, "float32").tflops


def test_laptop_parts_get_their_own_clock():
    laptop = _card("NVIDIA GeForce RTX 4060 Laptop GPU", (8, 9), 2370)

    peak = peak_tflops(laptop, 24, "bfloat16")

    assert peak.tflops == pytest.approx(24 * 2370e6 * 512 / 1e12)
    assert (peak.architecture, peak.unit) == ("Ada Lovelace", "16-bit tensor")


def test_bf16_before_ampere_has_no_peak():
    assert peak_tflops(_card("NVIDIA GeForce RTX 2080 Ti", (7, 5), 1545), 68,
                       "bfloat16") is None


@pytest.mark.parametrize("device, sms, dtype", [
    (None, 82, "bfloat16"),
    (_card("Some future GPU", (99, 0), 3000), 200, "bfloat16"),
    ({"name": "NVIDIA GeForce RTX 3090", "compute_capability": None}, 82, "bfloat16"),
    (_card("NVIDIA GeForce RTX 3090", (8, 6), None), 82, "bfloat16"),
    (_card("NVIDIA GeForce RTX 3090", (8, 6), 1695), None, "bfloat16"),
    (_card("NVIDIA GeForce RTX 3090", (8, 6), 1695), 82, None),
])
def test_anything_unknown_means_no_peak(device, sms, dtype):
    assert peak_tflops(device, sms, dtype) is None


def test_every_architecture_has_a_tensor_rate_above_its_fp32_rate():
    for arch in ARCHITECTURES.values():
        assert arch.tensor_16bit > arch.fp32


def test_mfu_counts_model_flops_not_the_recompute(telemetry_sample):
    import pandas as pd
    df = pd.DataFrame([telemetry_sample(i) for i in range(9)])
    summary = {"total_steps": 100, "elapsed_time_sec": 10.0, "tokens_per_step": 4096,
               "params_b": 1.5, "gradient_checkpointing": True,
               "compute_dtype": "bfloat16", "sm_count": 128,
               "measure_start": 100.0, "measure_end": 108.0}
    device = _card("NVIDIA GeForce RTX 4090", (8, 9), 2520)

    stats = metrics.compute_stats(df, summary, device=device)

    tokens_per_sec = 100 * 4096 / 10.0
    assert stats["mfu"] == pytest.approx(tokens_per_sec * 4 * 1.5e9 / (stats["peak_tflops"] * 1e12))
    assert stats["hfu"] == pytest.approx(stats["mfu"] * 6 / 4)
    assert stats["peak_basis"] == "Ada Lovelace, 16-bit tensor"


def test_no_device_info_no_mfu(telemetry_sample):
    import pandas as pd
    df = pd.DataFrame([telemetry_sample(i) for i in range(9)])
    summary = {"total_steps": 100, "elapsed_time_sec": 10.0, "tokens_per_step": 4096,
               "params_b": 1.5, "compute_dtype": "bfloat16", "sm_count": 128}

    assert metrics.compute_stats(df, summary)["mfu"] is None
