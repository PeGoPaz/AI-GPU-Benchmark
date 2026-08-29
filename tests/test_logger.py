"""BenchmarkLogger: run collection, export, and state-machine behaviour."""
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
    csv_path, png_path = BenchmarkLogger.export(df, output_dir=str(tmp_path))

    assert os.path.getsize(csv_path) > 0
    assert os.path.getsize(png_path) > 0
    assert csv_path.endswith(".csv")
    assert png_path.endswith(".png")


def test_export_csv_contains_telemetry_columns(telemetry_sample, tmp_path):
    df = _filled(telemetry_sample).stop()
    csv_path, _ = BenchmarkLogger.export(df, output_dir=str(tmp_path))

    header = open(csv_path).readline()
    for column in ("temp_gpu", "temp_memory", "temp_headroom_c", "power_w", "time_sec"):
        assert column in header
