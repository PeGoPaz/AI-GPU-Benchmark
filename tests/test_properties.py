"""Properties that must hold for any input, not just the examples elsewhere.

The example-based tests pin known values; these let Hypothesis search for
the input that breaks an invariant — an unsorted timestamp, a one-token
dataset, a scale value just above a round number.
"""
import math

import numpy as np
import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st

from gpu_bench.cli import resolve
from gpu_bench.core import metrics, thermal
from gpu_bench.core.trainer import MODELS, PRECISIONS, PRESETS, pack_tokens
from gpu_bench.ui.widgets.gauge import nice_ceiling
from gpu_bench.utils.plots import envelope

watts = st.floats(min_value=0, max_value=2000, allow_nan=False)


@st.composite
def power_curves(draw, min_size=2):
    """Strictly increasing timestamps with a power reading at each."""
    gaps = draw(st.lists(st.floats(min_value=1e-3, max_value=10), min_size=min_size - 1,
                         max_size=200))
    start = draw(st.floats(min_value=0, max_value=1e9))
    times = start + np.concatenate([[0.0], np.cumsum(gaps)])
    power = draw(st.lists(watts, min_size=len(times), max_size=len(times)))
    return times, np.asarray(power)


# --- energy -----------------------------------------------------------------

@given(curve=power_curves(), level=watts)
def test_constant_power_integrates_to_power_times_duration(curve, level):
    times, _ = curve
    energy = metrics.trapezoid_joules(times, np.full(len(times), level))

    assert math.isclose(energy, level * (times[-1] - times[0]), rel_tol=1e-9, abs_tol=1e-6)


@given(curve=power_curves(min_size=3), data=st.data())
def test_energy_adds_up_across_a_split(curve, data):
    times, power = curve
    k = data.draw(st.integers(min_value=1, max_value=len(times) - 2))

    whole = metrics.trapezoid_joules(times, power)
    parts = (metrics.trapezoid_joules(times[:k + 1], power[:k + 1])
             + metrics.trapezoid_joules(times[k:], power[k:]))
    assert math.isclose(whole, parts, rel_tol=1e-9, abs_tol=1e-6)


@given(curve=power_curves())
def test_energy_is_never_negative_and_bounded_by_the_peak(curve):
    times, power = curve
    energy = metrics.trapezoid_joules(times, power)

    assert 0 <= energy <= power.max() * (times[-1] - times[0]) * (1 + 1e-9) + 1e-6


@given(curve=power_curves(), start_mj=st.integers(0, 10**12),
       increments=st.data())
def test_the_cards_counter_wins_over_the_samples(curve, start_mj, increments):
    times, power = curve
    steps = increments.draw(st.lists(st.integers(0, 10**6), min_size=len(times) - 1,
                                     max_size=len(times) - 1))
    counter = start_mj + np.concatenate([[0], np.cumsum(steps)])
    df = pd.DataFrame({"timestamp": times, "power_w": power, "energy_mj": counter})

    energy, source = metrics.energy_joules(df)

    assert source == "counter"
    assert energy == (counter[-1] - counter[0]) / 1000


@given(curve=power_curves(), bounds=st.tuples(st.floats(0, 2e9), st.floats(0, 2e9)))
def test_the_timed_window_only_keeps_rows_inside_it(curve, bounds):
    times, power = curve
    start, end = sorted(bounds)
    df = pd.DataFrame({"timestamp": times, "power_w": power})

    window = metrics.timed_window(df, start, end)

    assert window["timestamp"].between(start, end).all()
    assert len(window) == int(((times >= start) & (times <= end)).sum())


@given(masks=st.lists(st.one_of(st.none(), st.integers(0, 2**16)), min_size=1, max_size=50),
       bits=st.integers(1, 2**16))
def test_throttled_share_is_a_percentage(masks, bits):
    df = pd.DataFrame({"throttle_reasons": pd.array(masks, dtype="Int64")})

    share = metrics.limited_share(df, bits)

    if all(m is None for m in masks):
        assert share is None
    else:
        assert share is not None and 0 <= share <= 100


@given(curve=power_curves())
def test_transient_figures_are_ordered(curve):
    times, power = curve
    assume(power.mean() > 0)
    stats = metrics.transient_stats(pd.DataFrame({"timestamp": times, "power_w": power}))

    assert stats["native_peak_power_w"] >= stats["native_p99_power_w"] - 1e-9
    assert stats["transient_ratio"] >= 1 - 1e-9


# --- drawing ----------------------------------------------------------------

@given(y=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=3000),
       max_points=st.integers(4, 500))
def test_the_envelope_keeps_every_extreme_and_the_order(y, max_points):
    x = np.arange(len(y), dtype=float)

    ex, ey = envelope(x, y, max_points)

    assert len(ey) <= max_points
    assert ey.max() == max(y) and ey.min() == min(y)
    assert np.all(np.diff(ex) > 0)


@given(st.floats(min_value=1e-6, max_value=1e9))
def test_scale_ceilings_are_round_and_never_below_the_value(value):
    ceiling = nice_ceiling(value)
    mantissa = ceiling / 10 ** math.floor(math.log10(ceiling))

    assert ceiling >= value
    assert ceiling <= 2 * value * (1 + 1e-9)
    assert any(math.isclose(mantissa, m) for m in (1, 2, 2.5, 5, 10))


# --- workload ---------------------------------------------------------------

@given(sequences=st.lists(st.lists(st.integers(0, 50_000), max_size=40), max_size=60),
       seq_len=st.integers(1, 64), separator=st.one_of(st.none(), st.integers(0, 50_000)))
def test_packing_cuts_exact_blocks_and_loses_only_the_tail(sequences, seq_len, separator):
    stream = [t for ids in sequences for t in ids + ([separator] if separator is not None
                                                     else [])]
    if len(stream) < seq_len:
        try:
            pack_tokens(sequences, seq_len, separator)
        except ValueError:
            return
        raise AssertionError("a dataset shorter than one block must be refused")

    blocks = pack_tokens(sequences, seq_len, separator)

    assert all(len(block) == seq_len for block in blocks)
    assert len(blocks) == len(stream) // seq_len
    assert [t for block in blocks for t in block] == stream[:len(blocks) * seq_len]


def _mangle(label: str, data) -> str:
    """The label as someone might type it: any case, separators swapped."""
    out = []
    for ch in label:
        if ch in " -_":
            out.append(data.draw(st.sampled_from([" ", "-", "_", ""])))
        elif ch in "()":
            out.append(data.draw(st.sampled_from([ch, ""])))
        else:
            out.append(data.draw(st.sampled_from([ch.lower(), ch.upper()])))
    return "".join(out)


@given(registry=st.sampled_from([MODELS, PRECISIONS, PRESETS]), data=st.data())
def test_every_entry_resolves_however_its_label_is_typed(registry, data):
    spec = data.draw(st.sampled_from(registry))

    assert resolve(registry, _mangle(spec.label, data), "entry") is spec


# --- thermal model ----------------------------------------------------------

@given(tau=st.floats(5, 200), start=st.floats(25, 60), rise=st.floats(5, 45),
       periods=st.floats(1, 6))
def test_the_heating_fit_recovers_any_clean_curve(tau, start, rise, periods):
    t = np.arange(0, tau * periods, 0.25)
    assume(len(t) >= thermal.MIN_POINTS)
    temps = start + rise * (1 - np.exp(-t / tau))
    assume(np.ptp(temps) >= thermal.MIN_RISE_C)

    fit = thermal.fit_heating(t, temps)

    assert fit is not None
    assert math.isclose(fit.tau_s, tau, rel_tol=0.02)
    assert math.isclose(fit.steady_c, start + rise, abs_tol=0.02 * rise + 0.1)
    assert fit.saturated == (t[-1] >= 3 * fit.tau_s)
