"""BenchmarkLogger: run collection, export, and state-machine behaviour."""
import json
import os

from app.utils.logger import BenchmarkLogger


def _filled(sample, n=5):
    lg = BenchmarkLogger()
    lg.start()
    for i in range(n):
        lg.log(sample(i))
    return lg


def test_stop_returns_collected_run(telemetry_sample):
    df = _filled(telemetry_sample).stop()
    assert df is not None
    assert len(df) == 5


def test_time_axis_starts_at_zero(telemetry_sample):
    df = _filled(telemetry_sample).stop()
    assert df["time_sec"].iloc[0] == 0.0
    assert df["time_sec"].is_monotonic_increasing


def test_empty_run_returns_none():
    lg = BenchmarkLogger()
    lg.start()
    assert lg.stop() is None


def test_log_ignored_while_stopped(telemetry_sample):
    lg = _filled(telemetry_sample)
    lg.stop()
    lg.log(telemetry_sample(99))
    assert len(lg.to_dataframe()) == 5


def test_start_clears_previous_run(telemetry_sample):
    lg = _filled(telemetry_sample)
    lg.stop()
    lg.start()
    assert lg.to_dataframe() is None


def test_to_dataframe_is_repeatable_and_read_only(telemetry_sample):
    """Regression: export used to call stop(), coupling it to run state."""
    lg = _filled(telemetry_sample)
    first, second = lg.to_dataframe(), lg.to_dataframe()

    assert len(first) == len(second) == 5
    assert lg.is_logging is True, "reading data must not stop collection"

    lg.log(telemetry_sample(5))
    assert len(lg.to_dataframe()) == 6, "collection must continue after a read"


def test_export_writes_csv_and_png(telemetry_sample, tmp_path):
    df = _filled(telemetry_sample).stop()
    paths = BenchmarkLogger.export(df, output_dir=str(tmp_path))
    csv_path, png_path, report_path, trace_path = paths.csv, paths.plot, paths.report, paths.power_trace

    assert report_path is None                  # no report asked for
    assert trace_path is None                   # no native trace either
    assert os.path.getsize(csv_path) > 0
    assert os.path.getsize(png_path) > 0
    assert csv_path.endswith(".csv")
    assert png_path.endswith(".png")


def test_export_csv_contains_telemetry_columns(telemetry_sample, tmp_path):
    df = _filled(telemetry_sample).stop()
    csv_path = BenchmarkLogger.export(df, output_dir=str(tmp_path)).csv

    header = open(csv_path).readline()
    for column in ("temp_gpu", "temp_memory", "temp_headroom_c", "power_w", "time_sec"):
        assert column in header


def test_exported_plot_survives_a_card_without_fans(telemetry_sample, tmp_path):
    """The fan series is optional; an all-None column must not break export."""
    lg = BenchmarkLogger()
    lg.start()
    for i in range(5):
        lg.log(telemetry_sample(i, fan_speed_pct=None, fan_rpm=None))

    png_path = BenchmarkLogger.export(lg.stop(), output_dir=str(tmp_path)).plot

    assert os.path.getsize(png_path) > 0



def test_report_is_written_beside_the_files_it_describes(telemetry_sample, tmp_path):
    df = _filled(telemetry_sample).stop()
    paths = BenchmarkLogger.export(
        df, output_dir=str(tmp_path), report={"schema_version": 1, "files": {}})
    csv_path, png_path, report_path = paths.csv, paths.plot, paths.report

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["files"] == {"csv": os.path.basename(csv_path),
                               "plot": os.path.basename(png_path)}
    assert os.path.dirname(report_path) == os.path.dirname(csv_path)



# --- native power trace ------------------------------------------------------------

def _chunk(start, n, watts=200.0):
    import numpy as np
    seconds = start + np.arange(n) * 0.002
    return seconds, np.full(n, watts, dtype=np.float32)


def test_power_trace_shares_the_telemetry_time_axis(telemetry_sample):
    lg = _filled(telemetry_sample)                # first row at timestamp 100.0
    lg.log_power_trace(_chunk(100.5, 3))
    lg.log_power_trace(_chunk(100.506, 2))

    trace = lg.power_trace()

    assert len(trace) == 5
    assert trace["time_sec"].iloc[0] == 0.5
    assert trace["time_sec"].is_monotonic_increasing


def test_power_trace_is_only_kept_while_logging(telemetry_sample):
    lg = _filled(telemetry_sample)
    lg.stop()
    lg.log_power_trace(_chunk(200.0, 10))

    assert lg.power_trace() is None


def test_new_run_starts_without_the_old_trace(telemetry_sample):
    lg = _filled(telemetry_sample)
    lg.log_power_trace(_chunk(100.0, 10))
    lg.stop()
    lg.start()

    assert lg.power_trace() is None



def test_export_writes_the_trace_beside_the_telemetry(telemetry_sample, tmp_path):
    lg = _filled(telemetry_sample)
    lg.log_power_trace(_chunk(100.5, 50, watts=275.0))
    df = lg.stop()

    paths = BenchmarkLogger.export(df, output_dir=str(tmp_path),
                                   report={"schema_version": 1, "files": {}},
                                   trace=lg.power_trace())

    assert os.path.basename(paths.power_trace).startswith("power_")
    header = open(paths.power_trace).readline().strip()
    assert header == "timestamp,power_w,time_sec"
    with open(paths.report, encoding="utf-8") as f:
        assert json.load(f)["files"]["power_trace"] == os.path.basename(paths.power_trace)


def test_export_with_a_summary_writes_the_html_page(telemetry_sample, tmp_path):
    from app.utils import report as report_module

    df = _filled(telemetry_sample).stop()
    summary = {"total_steps": 10, "requested_steps": 10, "aborted": False,
               "elapsed_time_sec": 4.0, "steps_per_sec": 2.5}
    rep = report_module.build_report(df, summary)

    paths = BenchmarkLogger.export(df, str(tmp_path), report=rep, summary=summary)

    assert paths.html.endswith(".html")
    with open(paths.html, encoding="utf-8") as f:
        assert f.read().startswith("<!doctype html>")
    with open(paths.report, encoding="utf-8") as f:
        assert json.load(f)["files"]["html"] == os.path.basename(paths.html)


def test_export_without_a_summary_writes_no_page(telemetry_sample, tmp_path):
    df = _filled(telemetry_sample).stop()

    paths = BenchmarkLogger.export(df, str(tmp_path), report={"schema_version": 1, "files": {}})

    assert paths.html is None
