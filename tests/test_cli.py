"""Headless mode: argument parsing and a full run with threadless workers."""
import io
import json

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from app import cli
from app.core import trainer as trainer_module


class FakeTelemetry(QObject):
    """Emits its snapshots synchronously from start() instead of polling."""

    data_updated = pyqtSignal(dict)
    gpu_name_ready = pyqtSignal(str)
    device_info_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    power_trace = pyqtSignal(object)
    sampler_status = pyqtSignal(dict)
    samples: list = []
    status = {"active": False, "detail": "native module not built"}
    trace = None      # a (seconds, watts) chunk to emit, when a test wants one

    def __init__(self, interval_ms=250, parent=None):
        super().__init__(parent)
        self.stopped = False

    def start(self):
        self.device_info_ready.emit({"name": "NVIDIA GeForce RTX 4060",
                                     "driver_version": "580.82",
                                     "cuda_driver_version": "13.0"})
        self.sampler_status.emit(self.status)
        for sample in self.samples:
            self.data_updated.emit(sample)
        if self.trace is not None:
            self.power_trace.emit(self.trace)

    def stop(self):
        self.stopped = True


class ScriptedTrainer(QObject):
    """Plays a whole run out of start(), ending in success or an error."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    loss_updated = pyqtSignal(int, float)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    max_steps_ready = pyqtSignal(int)
    outcome = "finish"
    total = 20

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.stopped = False
        self.joined = False

    def start(self):
        self.max_steps_ready.emit(self.total)
        for step in range(1, self.total + 1):
            self.progress_updated.emit(step)
            self.loss_updated.emit(step, 3.0 - step / 100)
        if self.outcome == "error":
            self.error_occurred.emit("HF Training Error: CUDA out of memory")
            return
        self.training_finished.emit({
            "model": self.kwargs["model"].label, "precision": self.kwargs["precision"].label,
            "batch_size": self.kwargs["batch_size"], "seq_len": self.kwargs["seq_len"],
            "total_steps": self.total, "requested_steps": self.total, "aborted": False,
            "elapsed_time_sec": 10.0, "steps_per_sec": 2.0,
            "tokens_per_step": 4096, "params_b": 1.54,
            "measure_start": 101.0, "measure_end": 104.0,
        })

    def stop(self):
        self.stopped = True

    def wait_for_exit(self, timeout_ms=30000):
        self.joined = True
        return True


def _parse(*argv):
    return cli.build_parser().parse_args(["--headless", *argv])


def _run(tmp_path, telemetry_sample, outcome="finish", total=20, samples=6, argv=()):
    FakeTelemetry.samples = [telemetry_sample(i) for i in range(samples)]
    ScriptedTrainer.outcome = outcome
    ScriptedTrainer.total = total
    out = io.StringIO()
    run = cli.HeadlessRun(cli.workload_from_args(_parse(*argv)), str(tmp_path), out=out,
                          trainer_cls=ScriptedTrainer, telemetry_cls=FakeTelemetry)
    codes = []
    run.done.connect(codes.append)
    run.start()
    return run, codes, out.getvalue()


# --- argument parsing -----------------------------------------------------------

def test_defaults_to_the_standard_preset():
    workload = cli.workload_from_args(_parse())

    assert workload["model"].label == "Qwen2.5 1.5B"
    assert workload["steps"] == 300
    assert workload["warmup_steps"] == 10


def test_flags_override_the_preset():
    workload = cli.workload_from_args(
        _parse("--preset", "stress", "--steps", "40", "--no-gradient-checkpointing"))

    assert workload["model"].label == "Qwen2.5 7B"          # from the preset
    assert workload["steps"] == 40                          # from the flag
    assert workload["gradient_checkpointing"] is False


@pytest.mark.parametrize("typed, label", [
    ("qwen2.5-7b", "Qwen2.5 7B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B"),
    ("phi-3.5-mini", "Phi-3.5 mini 3.8B"),
    ("TINYLLAMA_1.1B", "TinyLlama 1.1B"),
])
def test_models_can_be_named_the_way_people_type_them(typed, label):
    assert cli.resolve(trainer_module.MODELS, typed, "model").label == label


def test_int4_is_enough_to_pick_the_nf4_precision():
    assert cli.resolve(trainer_module.PRECISIONS, "int4", "precision").label == "INT4 (NF4)"


def test_ambiguous_model_is_refused_with_the_candidates(capsys):
    with pytest.raises(SystemExit):
        _parse("--model", "qwen2.5")

    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "Qwen2.5 1.5B" in err and "Qwen2.5 7B" in err


def test_unknown_model_lists_what_exists(capsys):
    with pytest.raises(SystemExit):
        _parse("--model", "gpt-5")

    assert "is not one of" in capsys.readouterr().err


def test_sequence_length_must_be_a_supported_one():
    with pytest.raises(SystemExit):
        _parse("--seq-len", "300")


def test_workload_flags_without_headless_are_flagged():
    args = cli.build_parser().parse_args(["--steps", "50"])

    assert "--steps" in cli.gui_only_misuse(args)
    assert cli.gui_only_misuse(_parse("--steps", "50")) is None


def test_list_prints_every_choice():
    out = io.StringIO()
    cli.print_registry(out)

    for spec in (*trainer_module.PRESETS, *trainer_module.MODELS, *trainer_module.PRECISIONS):
        assert spec.label in out.getvalue()


# --- a whole run ------------------------------------------------------------------

def test_finished_run_prints_the_summary_and_writes_everything(tmp_path, telemetry_sample):
    run, codes, out = _run(tmp_path, telemetry_sample)

    assert codes == [0]
    assert "=== Benchmark Summary ===" in out
    assert "Tokens/sec" in out
    written = sorted(p.suffix for p in tmp_path.iterdir())
    assert written == [".csv", ".json", ".png"]

    [report] = tmp_path.glob("report_*.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["gpu"]["driver_version"] == "580.82"
    assert len(data["loss"]["values"]) == 20


def test_threads_are_joined_before_exiting(tmp_path, telemetry_sample):
    """The trainer reports before freeing VRAM; exiting under it would kill
    the thread mid-cleanup."""
    run, _, _ = _run(tmp_path, telemetry_sample)

    assert run.trainer.joined
    assert run.telemetry.stopped


def test_progress_is_printed_about_twenty_times(tmp_path, telemetry_sample):
    _, _, out = _run(tmp_path, telemetry_sample, total=200)

    progress_lines = [line for line in out.splitlines() if line.startswith("Step ")]
    assert 20 <= len(progress_lines) <= 21
    assert progress_lines[-1].startswith("Step 200 / 200")


def test_failed_run_exits_nonzero_and_keeps_the_telemetry(tmp_path, telemetry_sample):
    _, codes, out = _run(tmp_path, telemetry_sample, outcome="error")

    assert codes == [1]
    assert "CUDA out of memory" in out
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".csv", ".png"]


def test_run_without_telemetry_is_a_failure(tmp_path, telemetry_sample):
    """No NVML, no numbers worth keeping — scripts must be able to tell."""
    _, codes, out = _run(tmp_path, telemetry_sample, samples=0)

    assert codes == [1]
    assert "nothing to export" in out


def test_ctrl_c_asks_the_trainer_to_stop(tmp_path, telemetry_sample):
    FakeTelemetry.samples = []
    run = cli.HeadlessRun(cli.workload_from_args(_parse()), str(tmp_path), out=io.StringIO(),
                          trainer_cls=ScriptedTrainer, telemetry_cls=FakeTelemetry)

    run.request_stop()

    assert run.trainer.stopped


def test_interrupted_run_still_writes_results_but_exits_130(tmp_path, telemetry_sample):
    FakeTelemetry.samples = [telemetry_sample(i) for i in range(6)]
    ScriptedTrainer.outcome = "finish"
    run = cli.HeadlessRun(cli.workload_from_args(_parse()), str(tmp_path), out=io.StringIO(),
                          trainer_cls=ScriptedTrainer, telemetry_cls=FakeTelemetry)
    codes = []
    run.done.connect(codes.append)
    run.request_stop()
    run.start()

    assert codes == [cli.EXIT_INTERRUPTED]
    assert any(tmp_path.glob("report_*.json"))


def test_trainer_receives_the_resolved_workload(tmp_path, telemetry_sample):
    run, _, _ = _run(tmp_path, telemetry_sample,
                     argv=("--model", "tinyllama", "--precision", "fp16", "--seq-len", "256"))

    kwargs = run.trainer.kwargs
    assert kwargs["model"].label == "TinyLlama 1.1B"
    assert kwargs["precision"].label == "FP16"
    assert kwargs["seq_len"] == 256


def test_main_lists_and_exits_without_a_window(monkeypatch, capsys):
    import main

    monkeypatch.setattr("sys.argv", ["main.py", "--list"])

    assert main.main() == 0
    assert "Presets:" in capsys.readouterr().out



def test_sampler_status_is_printed(tmp_path, telemetry_sample):
    _, _, out = _run(tmp_path, telemetry_sample)

    assert "Power sampler: native module not built" in out


def test_active_sampler_is_described_with_its_rate(tmp_path, telemetry_sample, monkeypatch):
    monkeypatch.setattr(FakeTelemetry, "status",
                        {"active": True, "detail": "instant power", "interval_us": 2000})
    _, _, out = _run(tmp_path, telemetry_sample)

    assert "Power sampler: native, instant power, every 2 ms" in out



def test_native_trace_reaches_the_summary_and_the_export(tmp_path, telemetry_sample,
                                                          monkeypatch):
    import numpy as np
    t = 101.0 + np.arange(500) * 0.002
    monkeypatch.setattr(FakeTelemetry, "trace",
                        (t, np.where(np.arange(500) % 50 == 0, 310.0, 205.0)))
    monkeypatch.setattr(FakeTelemetry, "status",
                        {"active": True, "detail": "instant power", "interval_us": 2000})

    _, codes, out = _run(tmp_path, telemetry_sample)

    assert codes == [0]
    assert "Peak Power, instant (W)" in out and "310.0" in out
    assert any(tmp_path.glob("power_*.csv"))
    [report] = tmp_path.glob("report_*.json")
    assert json.loads(report.read_text(encoding="utf-8"))["power_sampler"]["active"] is True
