"""How the card heats: a first-order thermal model fitted to the run.

Under a constant load a GPU and its cooler warm up like a capacitor
charging: quickly at first, then ever more slowly towards the temperature
where the cooler removes heat as fast as the chip makes it,

    T(t) = T∞ − (T∞ − T0) · e^(−t/τ)

τ, the thermal time constant, says how long that takes: 95 % of the way
after 3τ. It matters for a benchmark because clocks follow temperature — a
run shorter than about 3τ ends on a card that has not finished heating, and
its numbers flatter what the same card does under sustained load. The fit
says so, and predicts T∞ even when the run ended before reaching it.

Fitted with numpy alone: for a fixed τ the model is linear in T0 and T∞, so
every τ on a log-spaced grid gets a closed-form least-squares fit, and the
best one wins. Cheap, deterministic and free of starting-guess trouble.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below this many polls, or this rise in temperature, there is no curve to
# fit — NVML reports whole degrees, so a 1 °C rise is pure quantisation.
MIN_POINTS = 20
MIN_RISE_C = 3.0
# A fit explaining less of the variance than this is not a heating curve:
# a load that changed mid-run, a fan curve kicking in, a card already hot.
MIN_R_SQUARED = 0.9
# Utilisation that counts as the load being on: heating starts there, not at
# the warm-up boundary, which can sit well up the curve already.
LOAD_UTIL_PCT = 50

_GRID_POINTS = 400


@dataclass(frozen=True)
class ThermalFit:
    tau_s: float          # thermal time constant
    start_c: float        # T0, the fitted temperature when the load began
    steady_c: float       # T∞, where the card would settle under this load
    r_squared: float
    duration_s: float     # how long the fitted curve covers

    @property
    def saturated(self) -> bool:
        """Whether the run lasted the 3τ it takes to get 95 % of the way."""
        return self.duration_s >= 3 * self.tau_s

    @property
    def progress(self) -> float:
        """How far towards T∞ the card got, 0 to 1: 1 − e^(−duration/τ)."""
        return float(1 - np.exp(-self.duration_s / self.tau_s))


def fit_heating(seconds: np.ndarray, temps: np.ndarray) -> ThermalFit | None:
    """The best first-order fit to a heating curve, or None if there is none.

    None when there are too few points or too little rise, when the best τ
    lands on the edge of the search grid (an instant step, or a curve so slow
    it is indistinguishable from a straight line — T∞ would be a guess), when
    the curve falls instead of rising, and when the fit explains too little.
    """
    t = np.asarray(seconds, dtype=float)
    y = np.asarray(temps, dtype=float)
    if len(t) < MIN_POINTS or np.ptp(y) < MIN_RISE_C:
        return None
    t = t - t[0]
    duration = float(t[-1])
    if duration <= 0:
        return None

    taus = np.geomspace(duration / 100, duration * 10, _GRID_POINTS)
    x = 1 - np.exp(-t[None, :] / taus[:, None])          # one row per τ
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).sum(axis=1)
    rise = ((x - x_mean) * (y - y_mean)).sum(axis=1) / var   # T∞ − T0
    start = y_mean - rise * x_mean[:, 0]                      # T0
    residual = ((start[:, None] + rise[:, None] * x - y) ** 2).sum(axis=1)

    best = int(np.argmin(residual))
    if best in (0, _GRID_POINTS - 1) or rise[best] <= 0:
        return None
    r_squared = 1 - residual[best] / ((y - y_mean) ** 2).sum()
    if r_squared < MIN_R_SQUARED:
        return None
    return ThermalFit(tau_s=float(taus[best]), start_c=float(start[best]),
                      steady_c=float(start[best] + rise[best]),
                      r_squared=float(r_squared), duration_s=duration)


def fit_run(df: pd.DataFrame, end: float | None = None) -> ThermalFit | None:
    """The fit for a run's telemetry, from the moment the load came on.

    The curve starts at the first poll with the GPU busy rather than at the
    timed window: by the end of warm-up the card may already be halfway up
    it, and the early, steep part is what pins τ down. It ends with the
    timed window, before the cool-down while VRAM is freed.
    """
    if 'gpu_util_pct' not in df.columns or df.empty:
        return None
    busy = df[df['gpu_util_pct'] >= LOAD_UTIL_PCT]
    if busy.empty:
        return None
    rows = df[df['timestamp'] >= busy['timestamp'].iloc[0]]
    if end is not None:
        rows = rows[rows['timestamp'] <= end]
    return fit_heating(rows['timestamp'].to_numpy(dtype=float),
                       rows['temp_gpu'].to_numpy(dtype=float))
