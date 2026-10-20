"""The Prometheus endpoint: exposition format, units, and a live scrape."""
import re
import urllib.error
import urllib.request

import pytest

from gpu_bench.core.telemetry import THROTTLE_SW_POWER_CAP
from gpu_bench.utils.prometheus import (
    CONTENT_TYPE,
    NAMES,
    MetricsServer,
    MetricsState,
    escape_label,
)

# One sample line of the text format: name, optional {labels}, a number.
SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?P<labels>.*)\})? '
                    r'(?P<value>-?[0-9.e+-]+|NaN)$')


def _parse(text):
    """{(name, labels): value}, checking every line is HELP, TYPE or a sample."""
    samples, typed = {}, set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            assert kind in ("gauge", "counter")
            typed.add(name)
            continue
        match = SAMPLE.match(line)
        assert match, f"not a valid sample line: {line!r}"
        assert match["name"] in typed, f"{match['name']} has no TYPE line"
        samples[(match["name"], match["labels"] or "")] = float(match["value"])
    return samples


def test_snapshot_is_exported_in_base_units(telemetry_sample):
    state = MetricsState()
    state.update_telemetry(telemetry_sample(0, throttle_reasons=THROTTLE_SW_POWER_CAP))

    samples = _parse(state.render())

    assert samples[("gpu_bench_temperature_celsius", 'sensor="gpu"')] == 65
    assert samples[("gpu_bench_utilization_ratio", "")] == 0.98
    assert samples[("gpu_bench_memory_used_bytes", "")] == 5000 * 2**20
    assert samples[("gpu_bench_sm_clock_hertz", "")] == 2600e6
    assert samples[("gpu_bench_energy_joules_total", "")] == 1000.0
    assert samples[("gpu_bench_throttled", 'reason="power"')] == 1
    assert samples[("gpu_bench_throttled", 'reason="thermal"')] == 0


def test_missing_readings_are_left_out_not_zero(telemetry_sample):
    state = MetricsState()
    state.update_telemetry(telemetry_sample(0, fan_speed_pct=None, power_peak_w=None))

    names = {name for name, _ in _parse(state.render())}

    assert "gpu_bench_fan_speed_ratio" not in names
    assert "gpu_bench_power_peak_watts" not in names


def test_run_info_carries_the_workload_and_replaces_itself():
    state = MetricsState()
    state.set_run(None, "Qwen2.5 1.5B", "BF16", "0.1.0")
    state.set_run("NVIDIA GeForce RTX 4060", "Qwen2.5 1.5B", "BF16", "0.1.0")

    info = [k for k in _parse(state.render()) if k[0] == "gpu_bench_run_info"]

    assert info == [("gpu_bench_run_info", 'gpu="NVIDIA GeForce RTX 4060",model="Qwen2.5 1.5B",'
                                           'precision="BF16",version="0.1.0"')]


def test_label_values_are_escaped():
    assert escape_label('a "b" \\ c\nd') == 'a \\"b\\" \\\\ c\\nd'


def test_every_family_has_the_prefix_and_counters_end_in_total():
    from gpu_bench.utils.prometheus import FAMILIES

    for family in FAMILIES:
        assert ("gpu_bench_" + family.name) in NAMES
        assert family.name.endswith("_total") == (family.type == "counter")


def test_a_live_scrape(telemetry_sample):
    state = MetricsState()
    state.update_progress(42, 310, 8192.0)
    state.update_loss(2.5)
    server = MetricsServer(state, port=0)
    server.start()
    try:
        host, port = server.address
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=5) as response:
            body = response.read().decode()
            assert response.headers["Content-Type"] == CONTENT_TYPE
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://{host}:{port}/", timeout=5)
        assert missing.value.code == 404
    finally:
        server.stop()

    samples = _parse(body)
    assert samples[("gpu_bench_training_step", "")] == 42
    assert samples[("gpu_bench_training_tokens_per_second", "")] == 8192.0
    assert samples[("gpu_bench_training_loss", "")] == 2.5


def test_a_headless_run_feeds_the_metrics(tmp_path, telemetry_sample):
    import io

    from test_cli import FakeTelemetry, ScriptedTrainer
    from test_cli import _parse as parse_args

    from gpu_bench import cli

    FakeTelemetry.samples = [telemetry_sample(i) for i in range(6)]
    ScriptedTrainer.outcome, ScriptedTrainer.total = "finish", 20
    state = MetricsState()
    run = cli.HeadlessRun(cli.workload_from_args(parse_args()), str(tmp_path),
                          out=io.StringIO(), trainer_cls=ScriptedTrainer,
                          telemetry_cls=FakeTelemetry, metrics=state)
    run.start()

    samples = _parse(state.render())
    assert samples[("gpu_bench_training_step", "")] == 20
    assert samples[("gpu_bench_training_steps_planned", "")] == 20
    assert samples[("gpu_bench_temperature_celsius", 'sensor="gpu"')] == 70
    assert any(name == "gpu_bench_run_info" and 'gpu="NVIDIA GeForce RTX 4060"' in labels
               for name, labels in samples)


def test_metrics_port_needs_headless():
    from gpu_bench import cli

    args = cli.build_parser().parse_args(["--metrics-port", "9400"])

    assert "--metrics-port" in cli.gui_only_misuse(args)
