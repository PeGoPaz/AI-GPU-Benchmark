"""One JSON file per exported run: what ran, on what, and how fast.

The CSV holds the raw telemetry and the PNG the curves, but neither says
which card, driver, model or settings produced them — and a benchmark number
without that context cannot be compared with anything. The report is the
stable, versioned format later features (run comparison, multi-run averages)
read back, so fields are added rather than renamed, and schema_version goes up
when that rule has to be broken.
"""
import importlib.metadata
import platform
from datetime import datetime, timezone

from gpu_bench import __version__
from gpu_bench.core.metrics import compute_stats

REPORT_SCHEMA_VERSION = 1

# Distribution names, as pip knows them, of everything whose version can move
# a result. Read from package metadata rather than by importing: torch would
# cost seconds to import here, and a missing package is simply null.
_TRACKED_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "datasets",
                     "nvidia-ml-py", "PyQt6", "ai-gpu-benchmark-native")

_WORKLOAD_KEYS = ("model", "model_id", "precision", "batch_size", "seq_len",
                  "gradient_checkpointing", "warmup_steps", "requested_steps",
                  "tokens_per_step")

_RESULT_KEYS = ("total_steps", "aborted", "elapsed_time_sec", "steps_per_sec")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_report(df, summary: dict, device: dict | None = None,
                 loss: tuple[list[int], list[float]] | None = None,
                 trace=None, sampler: dict | None = None) -> dict:
    """Assembles the report for a finished run. File paths are added on export.

    `sampler` is the telemetry thread's description of how power was sampled;
    the transient figures it enables land in results alongside the rest.
    """
    steps, values = loss if loss is not None else ([], [])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": __version__,
        "gpu": device or {},
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "packages": package_versions(),
        "workload": {key: summary.get(key) for key in _WORKLOAD_KEYS},
        "results": {
            **{key: summary.get(key) for key in _RESULT_KEYS},
            **compute_stats(df, summary, trace),
        },
        "power_sampler": sampler or {"active": False, "detail": "unknown"},
        "loss": {"steps": list(steps), "values": list(values)},
        "files": {},
    }
