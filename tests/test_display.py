"""Headline cards: the five numbers a finished run is compared by."""
import numpy as np
import pandas as pd

from app.utils import display


def _df(sample, n=9):
    df = pd.DataFrame([sample(i) for i in range(n)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


FULL = {"total_steps": 100, "requested_steps": 100, "aborted": False,
        "elapsed_time_sec": 20.0, "steps_per_sec": 5.0, "tokens_per_step": 4096,
        "params_b": 1.5, "gradient_checkpointing": False,
        "measure_start": 102.0, "measure_end": 106.0}


def test_a_complete_run_gets_five_headlines(telemetry_sample):
    cards = display.headline_cards(_df(telemetry_sample), FULL)

    assert [c.label for c in cards] == ["Throughput", "Compute", "Efficiency",
                                        "Peak Power", "Max Temp"]
    assert (cards[0].value, cards[0].unit) == ("20,480", "tokens/s")
    assert cards[0].caption == "5.00 steps/s"
    assert cards[2].unit == "tokens/J"


def test_without_token_counts_throughput_falls_back_to_steps(telemetry_sample):
    summary = {"total_steps": 150, "requested_steps": 150, "aborted": False,
               "elapsed_time_sec": 70.0, "steps_per_sec": 2.14}

    cards = display.headline_cards(_df(telemetry_sample), summary)

    assert (cards[0].value, cards[0].unit) == ("2.14", "steps/s")
    assert "Compute" not in [c.label for c in cards]
    assert "Efficiency" not in [c.label for c in cards]


def test_peak_power_prefers_the_instant_trace(telemetry_sample):
    t = 101.0 + np.arange(1000) * 0.002
    trace = pd.DataFrame({"timestamp": t, "time_sec": t - 100.0,
                          "power_w": np.where(np.arange(1000) % 100 == 0, 330.0, 240.0)})

    cards = display.headline_cards(_df(telemetry_sample), FULL, trace)
    peak = next(c for c in cards if c.label == "Peak Power")

    assert peak.value == "330"
    assert peak.caption.startswith("instant")
