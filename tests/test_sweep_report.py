"""The sweep page: rebuilt from the sweep record alone."""
from gpu_bench.utils.sweep_report import build_sweep_html


def _metric(v, n=3, cv=0.01):
    return {"mean": v, "stdev": v * cv, "cv": cv, "min": v, "max": v, "n": n}


RECORD = {
    "schema_version": 1, "kind": "sweep", "created_at": "2026-10-20T12:00:00+00:00",
    "app_version": "0.1.0", "gpu": {"name": "NVIDIA GeForce RTX 4060 Laptop GPU"},
    "axes": {"batch_size": [1, 2, 4]}, "repeat": 3,
    "configurations": [
        {"label": "batch 1", "status": "ok",
         "metrics": {"tokens_per_sec": _metric(1000.0), "tokens_per_joule": _metric(40.0),
                     "max_temp_c": _metric(70.0)}},
        {"label": "batch 2", "status": "ok",
         "metrics": {"tokens_per_sec": _metric(1300.0), "tokens_per_joule": _metric(39.0),
                     "mfu": _metric(0.31), "max_temp_c": _metric(74.0)}},
        {"label": "batch 4", "status": "out of memory", "metrics": None},
    ],
    "skipped": [{"label": "batch 8", "reason": "weights need 9.1 GiB of 8.0 GiB"}],
    "front": [0, 1], "picks": {"fastest": 1, "most_efficient": 0, "balanced": 1},
}


def test_page_has_the_chart_the_table_and_the_skips():
    page = build_sweep_html(RECORD)

    assert "<h1>NVIDIA GeForce RTX 4060 Laptop GPU</h1>" in page
    assert "batch size 1, 2, 4 · 3 configurations, 3 run(s) each" in page
    assert page.count("<svg") == 1
    assert "1,300 ± 1.0 %" in page
    assert "front · fastest · balanced" in page
    assert "31.0%" in page
    assert "out of memory" in page
    assert "weights need 9.1 GiB of 8.0 GiB" in page


def test_nothing_finished_means_no_chart():
    record = {**RECORD, "configurations": [RECORD["configurations"][2]], "front": [],
              "picks": {}}

    page = build_sweep_html(record)

    assert "<svg" not in page
    assert "out of memory" in page


def test_labels_are_escaped():
    record = {**RECORD, "gpu": {"name": "<script>x</script>"}}

    assert "<script>x" not in build_sweep_html(record)
