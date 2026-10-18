"""High-rate power from the optional native module, with a quiet fallback.

The C++ extension in native/ samples the GPU's instantaneous power every
couple of milliseconds on a thread of its own. When it is built and NVML is
reachable, telemetry drains it on every 250 ms poll and gains the transients
those polls step over. When it is not — not built, no driver, a card without
a power reading — open_sampler() says why in one line and the app carries on
exactly as it did before the module existed.
"""
import importlib

# 2 ms: a 500 Hz trace is fine-grained enough to show per-step transients,
# at the cost of one NVML field read per interval on the native thread.
DEFAULT_INTERVAL_US = 2000

NOT_BUILT = "native module not built — power from 250 ms polls (pip install ./native)"


def open_sampler(device_index: int = 0, interval_us: int = DEFAULT_INTERVAL_US):
    """Returns (sampler, description) when native sampling works, else (None, why not).

    The sampler is created but not started; the caller owns its lifetime.
    """
    try:
        nvml_sampler = importlib.import_module("nvml_sampler")
    except ImportError:
        return None, NOT_BUILT

    try:
        sampler = nvml_sampler.PowerSampler(device_index=device_index,
                                            interval_us=interval_us)
    except (RuntimeError, ValueError) as e:
        return None, f"native sampler unavailable: {e}"
    return sampler, sampler.description


def describe_status(sampler, detail: str) -> dict:
    """What the UI, the CLI and the report say about how power was sampled."""
    status: dict = {"active": sampler is not None, "detail": detail}
    if sampler is not None:
        status["interval_us"] = sampler.interval_us
        try:
            status["module_version"] = importlib.import_module("nvml_sampler").__version__
        except ImportError:          # pragma: no cover - imported a moment ago
            pass
    return status
