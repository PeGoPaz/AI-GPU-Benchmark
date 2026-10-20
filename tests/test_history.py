"""The run history: SQLite, migrated by user_version, pointing at the report files."""
import json
import os
import sqlite3

import pytest

from gpu_bench.utils import history as history_module
from gpu_bench.utils.history import History, data_dir, record_run


def _report(tokens=12000.0, created="2026-10-20T10:00:00+00:00", gpu="NVIDIA GeForce RTX 4060"):
    return {"schema_version": 1, "created_at": created, "app_version": "0.1.0",
            "gpu": {"name": gpu},
            "workload": {"model": "Qwen2.5 1.5B", "precision": "BF16", "batch_size": 2,
                         "seq_len": 512, "gradient_checkpointing": True},
            "results": {"total_steps": 300, "aborted": False, "tokens_per_sec": tokens,
                        "tokens_per_joule": 40.0, "mfu": 0.3, "max_temp_c": 78.0}}


def _write(directory, name, report):
    path = directory / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_a_new_database_is_created_at_the_current_version(tmp_path):
    with History(str(tmp_path / "h.sqlite3")) as h:
        assert h.version == len(history_module.MIGRATIONS)
        assert h.runs() == []


def test_recording_a_run_keeps_its_figures_and_the_whole_report(tmp_path):
    with History(str(tmp_path / "h.sqlite3")) as h:
        h.record(str(tmp_path / "report_a.json"), _report())
        (row,) = h.runs()

        assert (row.gpu, row.model, row.batch_size, row.tokens_per_sec) == (
            "NVIDIA GeForce RTX 4060", "Qwen2.5 1.5B", 2, 12000.0)
        assert h.report(row.report_path)["results"]["mfu"] == 0.3


def test_recording_the_same_report_again_updates_it(tmp_path):
    path = str(tmp_path / "report_a.json")
    with History(str(tmp_path / "h.sqlite3")) as h:
        h.record(path, _report(tokens=1.0))
        h.record(path, _report(tokens=2.0))

        assert [r.tokens_per_sec for r in h.runs()] == [2.0]


def test_runs_come_newest_first_and_can_be_limited(tmp_path):
    with History(str(tmp_path / "h.sqlite3")) as h:
        for i, day in enumerate(("18", "20", "19")):
            h.record(str(tmp_path / f"report_{i}.json"),
                     _report(created=f"2026-10-{day}T10:00:00+00:00"))

        assert [r.created_at[8:10] for r in h.runs()] == ["20", "19", "18"]
        assert len(h.runs(limit=2)) == 2


def test_removing_a_run_leaves_its_file(tmp_path):
    path = _write(tmp_path, "report_a.json", _report())
    with History(str(tmp_path / "h.sqlite3")) as h:
        h.record(path, _report())
        h.remove(path)

        assert h.runs() == []
    assert (tmp_path / "report_a.json").exists()


def test_import_finds_reports_in_sweep_directories_and_skips_junk(tmp_path):
    logs = tmp_path / "logs"
    (logs / "sweep_1").mkdir(parents=True)
    _write(logs, "report_a.json", _report())
    _write(logs / "sweep_1", "report_b.json", _report())
    (logs / "report_broken.json").write_text("{not json", encoding="utf-8")
    _write(logs, "report_other.json", {"something": "else"})

    with History(str(tmp_path / "h.sqlite3")) as h:
        added, unreadable = h.import_dir(str(logs))

        assert added == 2
        assert len(unreadable) == 2
        assert len(h.runs()) == 2


def test_an_older_database_is_migrated_forward(tmp_path, monkeypatch):
    path = str(tmp_path / "h.sqlite3")
    with History(path) as h:
        h.record(str(tmp_path / "report_a.json"), _report())

    monkeypatch.setattr(history_module, "MIGRATIONS", [
        *history_module.MIGRATIONS, "ALTER TABLE runs ADD COLUMN note TEXT"])
    with History(path) as h:
        assert h.version == len(history_module.MIGRATIONS)
        assert len(h.runs()) == 1                               # data kept
        columns = [c[1] for c in h.db.execute("PRAGMA table_info(runs)")]
        assert "note" in columns


def test_a_failed_migration_leaves_the_database_as_it_was(tmp_path, monkeypatch):
    path = str(tmp_path / "h.sqlite3")
    History(path).close()
    monkeypatch.setattr(history_module, "MIGRATIONS", [
        *history_module.MIGRATIONS, "ALTER TABLE runs ADD COLUMN ok TEXT; NOT SQL"])

    with pytest.raises(sqlite3.Error):
        History(path)

    db = sqlite3.connect(path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    assert "ok" not in [c[1] for c in db.execute("PRAGMA table_info(runs)")]


def test_a_database_from_a_newer_app_is_refused(tmp_path):
    path = str(tmp_path / "h.sqlite3")
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 99")
    db.close()

    with pytest.raises(RuntimeError, match="newer"):
        History(path)


def test_the_history_lives_under_gpu_bench_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_BENCH_HOME", str(tmp_path / "home"))

    assert data_dir() == str(tmp_path / "home")
    assert record_run(str(tmp_path / "report_a.json"), _report()) is None
    assert (tmp_path / "home" / "history.sqlite3").exists()


def test_without_an_override_it_is_the_platforms_data_dir(monkeypatch):
    monkeypatch.delenv("GPU_BENCH_HOME")
    monkeypatch.setattr(history_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/data")

    assert data_dir() == os.path.join("/data", "ai-gpu-benchmark")


def test_a_broken_history_never_breaks_an_export(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("GPU_BENCH_HOME", str(blocker / "home"))

    assert record_run(str(tmp_path / "report_a.json"), _report()).startswith(
        "run not added to the history")
