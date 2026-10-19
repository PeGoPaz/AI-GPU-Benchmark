"""Run comparison: reports side by side, best values marked."""
import json

import pandas as pd
import pytest

from app.ui import compare_window as cw


def _write_run(directory, name, results, model="Qwen2.5 1.5B", precision="BF16",
               gpu="RTX 4060", telemetry=True, sample=None):
    files = {}
    if telemetry:
        df = pd.DataFrame([sample(i) for i in range(5)])
        df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
        df.to_csv(directory / f"run_{name}.csv", index=False)
        files["csv"] = f"run_{name}.csv"
    report = {"schema_version": 1, "gpu": {"name": gpu},
              "workload": {"model": model, "precision": precision},
              "results": results, "files": files}
    path = directory / f"report_{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_run_is_labelled_by_what_was_run_on_what(tmp_path, telemetry_sample):
    run = cw.load_run(_write_run(tmp_path, "a", {"tokens_per_sec": 1}, sample=telemetry_sample))

    assert run.label == "Qwen2.5 1.5B · BF16 · RTX 4060"
    assert len(run.telemetry) == 5


def test_missing_csv_keeps_the_numbers_and_says_why(tmp_path, telemetry_sample):
    path = _write_run(tmp_path, "a", {"tokens_per_sec": 1}, sample=telemetry_sample)
    (tmp_path / "run_a.csv").unlink()

    run = cw.load_run(path)

    assert run.telemetry is None
    assert "not found" in run.note


@pytest.mark.parametrize("content, reason", [("{ nope", "not valid JSON"),
                                             ('{"some": "thing"}', "not a run report")])
def test_unusable_files_are_rejected_with_a_reason(tmp_path, content, reason):
    path = tmp_path / "report_bad.json"
    path.write_text(content, encoding="utf-8")

    runs, errors = cw.load_runs([str(path)])

    assert runs == []
    assert reason in errors[0]


def test_best_value_follows_each_metrics_direction(tmp_path, telemetry_sample):
    a = cw.load_run(_write_run(tmp_path, "a", {"tokens_per_sec": 12000, "max_temp_c": 80,
                                               "avg_power_w": 150}, sample=telemetry_sample))
    b = cw.load_run(_write_run(tmp_path, "b", {"tokens_per_sec": 9000, "max_temp_c": 70,
                                               "avg_power_w": 110}, sample=telemetry_sample))

    rows = {label: (values, best) for label, values, best in cw.comparison_rows([a, b])}

    assert rows["Tokens/sec"] == (["12,000", "9,000"], 0)        # higher is better
    assert rows["Max temp (°C)"] == (["80", "70"], 1)            # lower is better
    assert rows["Avg power (W)"][1] is None                       # no right answer
    assert "Compute (est. TFLOPS)" not in rows                    # nobody reported it


def test_a_value_only_one_run_has_is_not_best(tmp_path, telemetry_sample):
    a = cw.load_run(_write_run(tmp_path, "a", {"tokens_per_sec": 12000}, sample=telemetry_sample))
    b = cw.load_run(_write_run(tmp_path, "b", {}, sample=telemetry_sample))

    [(label, values, best)] = cw.comparison_rows([a, b])

    assert values == ["12,000", "—"]
    assert best is None


@pytest.mark.ui
def test_window_shows_a_column_and_a_curve_per_run(qapp, tmp_path, telemetry_sample):
    from PyQt6.QtGui import QColor

    from app.ui.style import LIGHT

    runs = [cw.load_run(_write_run(tmp_path, n, {"tokens_per_sec": v}, sample=telemetry_sample))
            for n, v in (("a", 9000), ("b", 12000))]
    window = cw.CompareWindow(runs)

    assert window.table.columnCount() == 2
    assert len(window.curves["power"]) == 2
    best = window.table.item(0, 1)
    assert best.foreground().color() == QColor(LIGHT.ok)
    assert best.font().bold()


@pytest.mark.ui
def test_skipped_files_are_listed_above_the_table(qapp, tmp_path, telemetry_sample):
    runs = [cw.load_run(_write_run(tmp_path, "a", {"tokens_per_sec": 1}, sample=telemetry_sample))]
    window = cw.CompareWindow(runs, ["report_bad.json: not valid JSON"])

    assert not window.lbl_errors.isHidden()
    assert "report_bad.json" in window.lbl_errors.text()
