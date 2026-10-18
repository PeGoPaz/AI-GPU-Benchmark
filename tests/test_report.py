"""Run report: a versioned, self-describing JSON record of one run."""
import json

import pandas as pd

from app.utils import report


def _df(sample, n=9):
    df = pd.DataFrame([sample(i) for i in range(n)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


SUMMARY = {
    "model": "Qwen2.5 1.5B", "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "precision": "BF16", "batch_size": 2, "seq_len": 512,
    "gradient_checkpointing": True, "warmup_steps": 10, "requested_steps": 300,
    "tokens_per_step": 4096, "params_b": 1.54,
    "total_steps": 300, "aborted": False, "elapsed_time_sec": 120.0,
    "steps_per_sec": 2.5, "measure_start": 102.0, "measure_end": 106.0,
}

DEVICE = {"name": "NVIDIA GeForce RTX 4060", "driver_version": "580.82",
          "cuda_driver_version": "13.0", "vram_total_mb": 8188.0,
          "power_limit_w": 115.0}


def test_report_says_what_ran_and_on_what(telemetry_sample):
    r = report.build_report(_df(telemetry_sample), SUMMARY, DEVICE)

    assert r["schema_version"] == report.REPORT_SCHEMA_VERSION
    assert r["gpu"]["name"] == "NVIDIA GeForce RTX 4060"
    assert r["workload"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert r["workload"]["seq_len"] == 512
    assert r["results"]["steps_per_sec"] == 2.5
    assert r["results"]["tokens_per_sec"] == 300 * 4096 / 120.0


def test_report_survives_a_round_trip_through_json(telemetry_sample):
    """Everything in it must be plain JSON — a stray numpy scalar would only
    fail at the moment the user presses Export."""
    r = report.build_report(_df(telemetry_sample), SUMMARY, DEVICE,
                            loss=([1, 2], [3.1, 2.9]))

    assert json.loads(json.dumps(r)) == r


def test_report_records_the_loss_curve(telemetry_sample):
    r = report.build_report(_df(telemetry_sample), SUMMARY, loss=([1, 2], [3.1, 2.9]))

    assert r["loss"] == {"steps": [1, 2], "values": [3.1, 2.9]}


def test_missing_packages_are_null_not_an_error(telemetry_sample):
    """The suite stubs torch and friends, exactly like a broken install."""
    versions = report.build_report(_df(telemetry_sample), SUMMARY)["packages"]

    assert versions["torch"] is None or isinstance(versions["torch"], str)
    assert isinstance(versions["PyQt6"], str)


def test_report_without_a_gpu_reading_is_still_valid(telemetry_sample):
    r = report.build_report(_df(telemetry_sample), SUMMARY, device=None)

    assert r["gpu"] == {}



def test_report_records_how_power_was_sampled(telemetry_sample):
    import numpy as np
    t = 101.0 + np.arange(1000) * 0.002
    trace = pd.DataFrame({"timestamp": t, "time_sec": t - 100.0,
                          "power_w": np.where(np.arange(1000) % 50 == 0, 300.0, 200.0)})
    status = {"active": True, "detail": "instant power (NVML_FI_DEV_POWER_INSTANT)",
              "interval_us": 2000}

    r = report.build_report(_df(telemetry_sample), SUMMARY, DEVICE, trace=trace, sampler=status)

    assert r["power_sampler"] == status
    assert r["results"]["native_peak_power_w"] == 300.0
    assert json.loads(json.dumps(r)) == r


def test_report_without_the_native_sampler_says_so(telemetry_sample):
    r = report.build_report(_df(telemetry_sample), SUMMARY,
                            sampler={"active": False, "detail": "native module not built"})

    assert r["power_sampler"]["active"] is False
    assert r["results"]["native_peak_power_w"] is None
