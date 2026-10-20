"""Several runs of one workload, reduced to numbers with an error bar.

One run is an anecdote. Back to back, the same card on the same workload
still varies by a percent or two — clocks, temperature, what else the host
is doing — so a figure worth comparing is a mean with its spread, over runs
that agree with each other. A run that does not (a background update, a
thermal event) is dropped by a robust outlier test before averaging rather
than allowed to drag the mean.

Qt-free and file-free: it takes the results dicts the run reports carry and
returns plain dicts, so the CLI, the tests and any later UI share it.
"""
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# The results keys a series summarises, with how each reads and which way
# is better (+1 higher, −1 lower, 0 neither).
METRICS: tuple[tuple[str, str, str, int], ...] = (
    ("tokens_per_sec", "Tokens/sec", "{:,.0f}", 1),
    ("tokens_per_joule", "Efficiency (tokens/J)", "{:.1f}", 1),
    ("est_tflops", "Compute (est. TFLOPS)", "{:.1f}", 1),
    ("mfu", "MFU", "{:.1%}", 1),
    ("step_time_p50_s", "Step time p50 (s)", "{:.3f}", -1),
    ("avg_power_w", "Avg power (W)", "{:.0f}", 0),
    ("max_temp_c", "Max temp (°C)", "{:.0f}", -1),
)

# The metric outliers are judged by: the one the series exists to measure.
OUTLIER_KEY = "tokens_per_sec"
# Iglewicz and Hoaglin's cut-off for the modified z-score.
OUTLIER_Z = 3.5
# With fewer runs than this the "typical" run is not known well enough to
# call any of them atypical.
MIN_RUNS_FOR_OUTLIERS = 3


@dataclass(frozen=True)
class Aggregate:
    mean: float
    stdev: float          # sample standard deviation; 0 for a single run
    cv: float | None      # stdev / mean
    min: float
    max: float
    n: int


def modified_z_scores(values: Sequence[float]) -> np.ndarray:
    """How far each value sits from the median, in robust standard deviations.

    Built on the median absolute deviation, so one wild value cannot hide
    itself by inflating the spread it is measured against — the failure of
    the ordinary z-score on small samples. When more than half the values
    are identical the MAD is zero, and the mean absolute deviation stands
    in (scaled to match), as in IBM's formulation.
    """
    x = np.asarray(values, dtype=float)
    offset = x - float(np.median(x))
    mad = float(np.median(np.abs(offset)))
    if mad > 0:
        return np.asarray(0.6745 * offset / mad)
    mean_ad = float(np.abs(offset).mean())
    if mean_ad > 0:
        return np.asarray(offset / (1.253314 * mean_ad))
    return np.zeros(len(x))


def outliers(values: Sequence[float | None]) -> list[bool]:
    """Which runs to drop: |modified z| above the cut-off, given enough runs."""
    known = [(i, v) for i, v in enumerate(values) if v is not None]
    flags = [False] * len(values)
    if len(known) < MIN_RUNS_FOR_OUTLIERS:
        return flags
    scores = modified_z_scores([v for _, v in known])
    for (i, _), score in zip(known, scores, strict=True):
        flags[i] = bool(abs(score) > OUTLIER_Z)
    return flags


def aggregate(values: Sequence[float]) -> Aggregate | None:
    """Mean, sample standard deviation and range of one metric's values."""
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(x) == 0:
        return None
    mean = float(x.mean())
    stdev = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return Aggregate(mean=mean, stdev=stdev, cv=stdev / mean if mean else None,
                     min=float(x.min()), max=float(x.max()), n=len(x))


def summarise(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The series' figures from each run's results dict.

    Returns which runs were dropped (by index) and an aggregate per metric
    over the rest. A dropped run is dropped from every metric, not just the
    one it was judged on — its other numbers came from the same bad run.
    """
    dropped = outliers([r.get(OUTLIER_KEY) for r in results])
    kept = [r for r, drop in zip(results, dropped, strict=True) if not drop]
    metrics: dict[str, Any] = {}
    for key, *_ in METRICS:
        agg = aggregate([r[key] for r in kept if r.get(key) is not None])
        metrics[key] = asdict(agg) if agg is not None else None
    return {
        "runs": len(results),
        "dropped": [i for i, drop in enumerate(dropped) if drop],
        "metrics": metrics,
    }


def format_aggregate(agg: dict[str, Any], fmt: str) -> str:
    """'12,345 ± 1.2 %  (12,100 – 12,500)', or just the value for one run."""
    value = fmt.format(agg["mean"])
    if agg["n"] < 2:
        return value
    spread = f" ± {agg['cv'] * 100:.1f} %" if agg["cv"] is not None else ""
    return f"{value}{spread}  ({fmt.format(agg['min'])} – {fmt.format(agg['max'])})"


# How the picks read, wherever they are shown.
PICK_LABELS = {"fastest": "fastest", "most_efficient": "most efficient",
               "balanced": "balanced"}


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points no other point beats on both axes, both maximised.

    A configuration is on the front when nothing else is at least as fast
    and at least as efficient, and strictly better at one of them: every
    other configuration is a worse deal than some point on it. O(n²), which
    for the dozens of configurations a sweep runs is nothing.
    """
    front = []
    for i, (x, y) in enumerate(points):
        if not any(ox >= x and oy >= y and (ox > x or oy > y)
                   for j, (ox, oy) in enumerate(points) if j != i):
            front.append(i)
    return front


def picks(points: Sequence[tuple[float, float]]) -> dict[str, int]:
    """The three configurations worth naming: fastest, most efficient, balanced.

    Balanced is the front point nearest the ideal one that would be both the
    fastest and the most efficient, with each axis scaled to the range the
    sweep covered so that neither unit dominates the distance. Empty for no
    points.
    """
    if not points:
        return {}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def scaled(v: float, lo: float, hi: float) -> float:
        return (v - lo) / (hi - lo) if hi > lo else 1.0

    front = pareto_front(points)
    balanced = min(front, key=lambda i: math.hypot(1 - scaled(xs[i], min(xs), max(xs)),
                                                   1 - scaled(ys[i], min(ys), max(ys))))
    # On ties, max() keeps the first: the configuration listed earlier.
    return {
        "fastest": max(front, key=lambda i: xs[i]),
        "most_efficient": max(front, key=lambda i: ys[i]),
        "balanced": balanced,
    }


def build_record(reports: Sequence[dict[str, Any]], files: Sequence[str]) -> dict[str, Any]:
    """The series as one JSON document: what ran, each run's figures, the aggregate.

    `files` are the member reports' names, in run order, so the record can
    be traced back to every run behind it.
    """
    from datetime import datetime, timezone

    from gpu_bench import __version__

    first = reports[0] if reports else {}
    return {
        "schema_version": 1,
        "kind": "series",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": __version__,
        "gpu": first.get("gpu") or {},
        "workload": first.get("workload") or {},
        "members": [{"report": name,
                     **{key: (r.get("results") or {}).get(key) for key, *_ in METRICS}}
                    for r, name in zip(reports, files, strict=True)],
        **summarise([r.get("results") or {} for r in reports]),
    }
