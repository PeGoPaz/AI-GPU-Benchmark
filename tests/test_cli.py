"""Headless mode: argument parsing and a full run with threadless workers."""
import io
import json

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from gpu_bench import cli
from gpu_bench.core import trainer as trainer_module


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
    assert written == [".csv", ".html", ".json", ".png"]

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


# --- series (--repeat) ------------------------------------------------------------

def _run_series(qapp, tmp_path, telemetry_sample, repeat=3, outcomes=None, stop_after=None,
                **series_kwargs):
    """A Series of threadless runs, driven on a local event loop until done."""
    from PyQt6.QtCore import QEventLoop, QTimer

    from gpu_bench.batch import Cooldown, Series

    FakeTelemetry.samples = [telemetry_sample(i) for i in range(6)]
    ScriptedTrainer.total = 20
    outcomes = list(outcomes or ["finish"] * repeat)
    out = io.StringIO()
    made = []

    def make_run():
        ScriptedTrainer.outcome = outcomes[len(made)]
        run = cli.HeadlessRun(cli.workload_from_args(_parse()), str(tmp_path), out=out,
                              trainer_cls=ScriptedTrainer, telemetry_cls=FakeTelemetry)
        made.append(run)
        if stop_after is not None and len(made) == stop_after:
            run.start = lambda r=run: (series.request_stop(), cli.HeadlessRun.start(r))
        return run

    cooldown = None
    if series_kwargs:
        cooldown = Cooldown(series_kwargs.pop("cooldown_s"), out, **series_kwargs)
    series = Series(make_run, repeat, str(tmp_path), out=out, cooldown=cooldown)
    codes = []
    loop = QEventLoop()
    series.done.connect(lambda code: (codes.append(code), loop.quit()))
    QTimer.singleShot(0, series.start)
    QTimer.singleShot(5000, loop.quit)                # a hung series fails, not hangs
    loop.exec()
    return series, codes, out.getvalue(), made


def test_repeat_runs_the_workload_n_times_and_writes_the_series(qapp, tmp_path,
                                                                   telemetry_sample):
    series, codes, out, made = _run_series(qapp, tmp_path, telemetry_sample, repeat=3)

    assert codes == [0]
    assert len(made) == 3
    assert "=== Run 3 of 3 ===" in out
    assert "=== Series: 3 runs ===" in out
    record = json.loads(open(series.path, encoding="utf-8").read())
    assert record["kind"] == "series"
    assert len(record["members"]) == 3
    assert len({m["report"] for m in record["members"]}) == 3     # no file reused
    assert record["metrics"]["tokens_per_sec"]["n"] == 3


def test_a_failed_run_ends_the_series(qapp, tmp_path, telemetry_sample):
    series, codes, out, made = _run_series(qapp, tmp_path, telemetry_sample, repeat=3,
                                           outcomes=["finish", "error", "finish"])

    assert codes == [1]
    assert len(made) == 2
    assert "Fewer than two runs finished" in out
    assert series.path is None


def test_ctrl_c_finishes_the_current_run_and_summarises(qapp, tmp_path, telemetry_sample):
    series, codes, out, made = _run_series(qapp, tmp_path, telemetry_sample, repeat=4,
                                           stop_after=2)

    assert codes == [cli.EXIT_INTERRUPTED]
    assert len(made) == 2
    assert "=== Series: 2 runs ===" in out


def test_repeat_without_headless_is_pointed_out():
    args = cli.build_parser().parse_args(["--repeat", "3"])

    assert "--repeat" in cli.gui_only_misuse(args)


def test_repeat_must_be_positive(capsys):
    with pytest.raises(SystemExit):
        _parse("--repeat", "0")



class _Thermometer:
    """Plays back temperatures, one per reading; the last one repeats."""

    def __init__(self, *temps):
        self.temps, self.reads = list(temps), 0

    def __call__(self):
        self.reads += 1
        return self.temps.pop(0) if len(self.temps) > 1 else self.temps[0]


def test_cooldown_waits_for_the_starting_temperature(qapp, tmp_path, telemetry_sample):
    # baseline 45; after run 1: 70, 60, 50, 46 (cooled); after run 2: 46 at once
    thermometer = _Thermometer(45.0, 70.0, 60.0, 50.0, 46.0, 46.0)

    series, codes, out, made = _run_series(qapp, tmp_path, telemetry_sample, repeat=3,
                                           cooldown_s=60, read_temperature=thermometer,
                                           poll_ms=1)

    assert codes == [0] and len(made) == 3
    assert "Starting temperature: 45 °C" in out
    assert "Cooling down from 70 °C towards 45 °C..." in out
    assert "Cooled to 46 °C" in out


def test_cooldown_gives_up_after_its_limit(qapp, tmp_path, telemetry_sample):
    ticks = iter(range(0, 10_000, 10))       # ten seconds per reading

    series, codes, out, made = _run_series(
        qapp, tmp_path, telemetry_sample, repeat=2, cooldown_s=30,
        read_temperature=_Thermometer(45.0, 80.0), clock=lambda: next(ticks), poll_ms=1)

    assert codes == [0] and len(made) == 2
    assert "Still at 80 °C after 30 s; starting anyway." in out


def test_no_temperature_means_no_cooldown(qapp, tmp_path, telemetry_sample):
    series, codes, out, made = _run_series(qapp, tmp_path, telemetry_sample, repeat=2,
                                           cooldown_s=60, read_temperature=lambda: None)

    assert codes == [0] and len(made) == 2
    assert "no cool-down between runs" in out


def test_gpu_temperature_without_nvml_is_none(monkeypatch):
    from gpu_bench.core import telemetry

    def no_driver():
        raise Exception("NVML_ERROR_LIBRARY_NOT_FOUND")

    monkeypatch.setattr(telemetry.pynvml, "nvmlInit", no_driver)

    assert telemetry.read_gpu_temperature() is None


# --- sweeps (--sweep) -------------------------------------------------------------

def test_sweep_axes_parse_like_the_flags():
    args = _parse("--sweep", "batch-size=1,2,4", "seq_len=256,512", "precision=bf16,int4",
                  "checkpointing=on,off")

    assert args.sweep["batch_size"] == [1, 2, 4]
    assert args.sweep["seq_len"] == [256, 512]
    assert [p.label for p in args.sweep["precision"]] == ["BF16", "INT4 (NF4)"]
    assert args.sweep["gradient_checkpointing"] == [True, False]


@pytest.mark.parametrize("bad", ["batch-size", "colour=red", "seq-len=300", "batch=0",
                                 "checkpointing=maybe", "model=qwen2.5", "batch="])
def test_bad_sweep_axes_are_refused(bad, capsys):
    with pytest.raises(SystemExit):
        _parse("--sweep", bad)


def test_an_axis_cannot_be_swept_twice(capsys):
    with pytest.raises(SystemExit):
        _parse("--sweep", "batch=1,2", "batch-size=4")

    assert "swept twice" in capsys.readouterr().err


def test_grid_varies_the_last_axis_fastest():
    from gpu_bench.batch import grid

    configs = grid({"steps": 10}, {"batch_size": [1, 2], "seq_len": [256, 512]})

    assert [(c["batch_size"], c["seq_len"]) for c in configs] == [
        (1, 256), (1, 512), (2, 256), (2, 512)]
    assert all(c["steps"] == 10 for c in configs)


def test_what_cannot_fit_is_skipped_before_running():
    seven_b = cli.resolve(trainer_module.MODELS, "qwen2.5-7b", "model")
    configs = [{"model": seven_b, "precision": p} for p in
               (cli.resolve(trainer_module.PRECISIONS, "bf16", "precision"),
                cli.resolve(trainer_module.PRECISIONS, "int4", "precision"))]

    runnable, skipped = cli.split_by_fit(configs, vram_total_mib=8188)

    assert [c["precision"].label for c in runnable] == ["INT4 (NF4)"]
    assert "GiB of 8.0 GiB" in skipped[0][1]
    assert cli.split_by_fit(configs, None) == (configs, [])


def _run_sweep(tmp_path, telemetry_sample, axes, oom_batch=None, stop_in=None, repeat=1):
    from PyQt6.QtCore import QEventLoop, QTimer

    from gpu_bench.batch import Sweep, grid

    FakeTelemetry.samples = [telemetry_sample(i) for i in range(6)]
    ScriptedTrainer.total = 20
    out = io.StringIO()
    made = []

    def make_run(workload, output_dir):
        ScriptedTrainer.outcome = "error" if workload["batch_size"] == oom_batch else "finish"
        run = cli.HeadlessRun(workload, output_dir, out=out, trainer_cls=ScriptedTrainer,
                              telemetry_cls=FakeTelemetry)
        made.append(run)
        if stop_in is not None and len(made) == stop_in:
            run.start = lambda r=run: (sweep.request_stop(), cli.HeadlessRun.start(r))
        return run

    configs = grid(cli.workload_from_args(_parse()), axes)
    sweep = Sweep(configs, list(axes), make_run, repeat, str(tmp_path), out=out)
    codes = []
    loop = QEventLoop()
    sweep.done.connect(lambda code: (codes.append(code), loop.quit()))
    QTimer.singleShot(0, sweep.start)
    QTimer.singleShot(10000, loop.quit)
    loop.exec()
    return sweep, codes, out.getvalue(), made


def test_sweep_runs_every_configuration_into_its_own_directory(qapp, tmp_path,
                                                                 telemetry_sample):
    sweep, codes, out, made = _run_sweep(tmp_path, telemetry_sample,
                                         {"batch_size": [1, 2], "seq_len": [256, 512]})

    assert codes == [0]
    assert len(made) == 4
    assert "Configuration 4 of 4: batch 2, 512 tokens" in out
    assert [r["status"] for r in sweep.results] == ["ok"] * 4
    assert sweep.output_dir.startswith(str(tmp_path / "sweep_"))
    assert len(list((tmp_path / sweep.output_dir).glob("report_*.json"))) == 4


def test_out_of_memory_marks_the_configuration_and_the_sweep_goes_on(qapp, tmp_path,
                                                                       telemetry_sample):
    sweep, codes, out, made = _run_sweep(tmp_path, telemetry_sample,
                                         {"batch_size": [1, 8, 2]}, oom_batch=8)

    assert codes == [0]
    assert [r["status"] for r in sweep.results] == ["ok", "out of memory", "ok"]
    assert "batch 8" in out and "out of memory" in out


def test_each_configuration_is_a_series_with_repeat(qapp, tmp_path, telemetry_sample):
    sweep, codes, out, made = _run_sweep(tmp_path, telemetry_sample,
                                         {"batch_size": [1, 2]}, repeat=2)

    assert len(made) == 4
    assert all(r["record"]["metrics"]["tokens_per_sec"]["n"] == 2 for r in sweep.results)


def test_ctrl_c_ends_the_sweep_after_the_current_run(qapp, tmp_path, telemetry_sample):
    sweep, codes, out, made = _run_sweep(tmp_path, telemetry_sample,
                                         {"batch_size": [1, 2, 4]}, stop_in=2)

    assert codes == [cli.EXIT_INTERRUPTED]
    assert len(made) == 2
    assert "2 of 3 configurations run" in out


def test_a_sweep_where_nothing_finishes_fails(qapp, tmp_path, telemetry_sample):
    sweep, codes, out, made = _run_sweep(tmp_path, telemetry_sample,
                                         {"batch_size": [8]}, oom_batch=8)

    assert codes == [1]


def test_sweep_table_marks_the_front_and_names_the_picks():
    from gpu_bench.batch import Sweep

    def entry(batch, speed, efficiency):
        metric = lambda v: {"mean": v, "stdev": 0.0, "cv": 0.0, "min": v, "max": v, "n": 1}  # noqa: E731
        return {"workload": {"batch_size": batch}, "status": "ok", "files": [],
                "record": {"metrics": {"tokens_per_sec": metric(speed),
                                       "tokens_per_joule": metric(efficiency),
                                       "max_temp_c": metric(70.0)}}}

    out = io.StringIO()
    sweep = Sweep([{"batch_size": b} for b in (1, 2, 4, 8)], ["batch_size"],
                  lambda w, d: None, out=out)
    sweep.results = [entry(1, 1000, 40), entry(2, 1300, 39), entry(4, 1400, 30),
                     {"workload": {"batch_size": 8}, "status": "out of memory",
                      "record": None, "files": []}]
    sweep._conclude()
    text = out.getvalue()

    assert "front · most efficient" in text
    assert "front · fastest" in text
    assert "front · balanced" in text
    assert sweep.front == [0, 1, 2]
    assert "Balanced:        batch 2 — 1,300 tokens/s at 39.0 tokens/J" in text
