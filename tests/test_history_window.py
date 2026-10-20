"""View → History: the table over the run history database."""
import json

import pytest
from PyQt6.QtCore import Qt

from gpu_bench.ui import history_window as hw
from gpu_bench.utils.history import History

pytestmark = pytest.mark.ui


def _report(gpu, model, tokens, created):
    return {"created_at": created, "gpu": {"name": gpu},
            "workload": {"model": model, "precision": "BF16", "batch_size": 2, "seq_len": 512},
            "results": {"tokens_per_sec": tokens,
                        "tokens_per_joule": tokens / 300 if tokens else None,
                        "max_temp_c": 75.0}}


@pytest.fixture
def filled(tmp_path):
    """Three runs on file and in the history: two cards, one with an HTML page."""
    entries = [("NVIDIA GeForce RTX 4060", "Qwen2.5 1.5B", 9800.0, "2026-10-18T10:00:00"),
               ("NVIDIA GeForce RTX 4060", "TinyLlama 1.1B", 12000.0, "2026-10-19T10:00:00"),
               ("NVIDIA GeForce RTX 3070", "Qwen2.5 1.5B", None, "2026-10-20T10:00:00")]
    with History() as history:
        for i, entry in enumerate(entries):
            path = tmp_path / f"report_{i}.json"
            path.write_text(json.dumps(_report(*entry)), encoding="utf-8")
            history.record(str(path), _report(*entry))
    (tmp_path / "report_1.html").write_text("<html></html>", encoding="utf-8")
    return tmp_path


def _cell(window, row, column):
    return window.proxy.index(row, column).data()


def _column(name):
    return [header for header, _, _ in hw.COLUMNS].index(name)


def _select(window, *rows):
    selection = window.table.selectionModel()
    for row in rows:
        selection.select(window.proxy.index(row, 0),
                         selection.SelectionFlag.Select | selection.SelectionFlag.Rows)


def test_lists_every_run_newest_first(qapp, filled):
    window = hw.HistoryWindow()

    assert window.proxy.rowCount() == 3
    assert _cell(window, 0, _column("GPU")) == "NVIDIA GeForce RTX 3070"
    assert _cell(window, 0, _column("Tokens/s")) == "—"
    assert _cell(window, 2, _column("Tokens/s")) == "9,800"
    assert "3 runs" in window.lbl_status.text()


def test_numbers_sort_as_numbers_with_missing_values_last(qapp, filled):
    window = hw.HistoryWindow()

    window.table.sortByColumn(_column("Tokens/s"), Qt.SortOrder.DescendingOrder)

    assert [_cell(window, r, _column("Tokens/s")) for r in range(3)] == ["12,000", "9,800", "—"]


def test_the_filter_matches_any_column(qapp, filled):
    window = hw.HistoryWindow()

    window.filter.setText("3070")
    assert window.proxy.rowCount() == 1
    window.filter.setText("qwen")
    assert window.proxy.rowCount() == 2


def test_buttons_follow_the_selection(qapp, filled):
    window = hw.HistoryWindow(compare=lambda paths: None)
    assert not window.btn_open.isEnabled() and not window.btn_remove.isEnabled()

    _select(window, 0)
    assert window.btn_open.isEnabled() and not window.btn_compare.isEnabled()

    _select(window, 1)
    assert window.btn_compare.isEnabled() and not window.btn_open.isEnabled()


def test_compare_hands_over_the_selected_reports(qapp, filled):
    compared = []
    window = hw.HistoryWindow(compare=compared.append)
    window.filter.setText("4060")

    _select(window, 0, 1)
    window.compare_selected()

    assert len(compared) == 1
    assert sorted(p.rsplit("_", 1)[-1] for p in compared[0]) == ["0.json", "1.json"]


def test_remove_forgets_the_run_but_keeps_its_file(qapp, filled):
    window = hw.HistoryWindow()

    _select(window, 0)
    window.remove_selected()

    assert window.proxy.rowCount() == 2
    assert (filled / "report_2.json").exists()


def test_open_prefers_the_html_page(qapp, filled, monkeypatch):
    opened = []
    monkeypatch.setattr(hw.QDesktopServices, "openUrl", lambda url: opened.append(url))
    window = hw.HistoryWindow()

    window.open_report(1)            # the TinyLlama run, which has a page
    window.open_report(2)            # the oldest, JSON only

    assert opened[0].toLocalFile().endswith("report_1.html")
    assert opened[1].toLocalFile().endswith("report_0.json")


def test_a_moved_report_is_explained_not_opened(qapp, filled, monkeypatch):
    opened = []
    monkeypatch.setattr(hw.QDesktopServices, "openUrl", lambda url: opened.append(url))
    (filled / "report_2.json").unlink()
    window = hw.HistoryWindow()

    window.open_report(0)

    assert opened == []
    assert "no longer exists" in window.lbl_status.text()


def test_import_folder_adds_reports_and_refreshes(qapp, tmp_path):
    (tmp_path / "logs").mkdir()
    report = _report("NVIDIA GeForce RTX 4090", "Qwen2.5 3B", 20000.0, "2026-10-20T11:00:00")
    (tmp_path / "logs" / "report_x.json").write_text(json.dumps(report), encoding="utf-8")
    window = hw.HistoryWindow()

    window.import_folder(str(tmp_path / "logs"))

    assert window.proxy.rowCount() == 1
    assert "Imported 1 runs" in window.lbl_status.text()
