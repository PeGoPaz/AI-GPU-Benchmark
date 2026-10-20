"""Series statistics: robust outliers, mean and spread, the JSON record."""
import pytest

from gpu_bench.core import series


def test_a_wild_run_is_flagged_among_steady_ones():
    assert series.outliers([1000, 1010, 995, 1005, 700]) == [False] * 4 + [True]


def test_ordinary_spread_flags_nothing():
    assert series.outliers([1000, 1020, 980, 1010, 990]) == [False] * 5


def test_two_runs_are_too_few_to_call_either_an_outlier():
    assert series.outliers([1000, 500]) == [False, False]


def test_identical_runs_with_one_exception_still_find_it():
    """MAD is zero when most values agree; the mean deviation takes over."""
    assert series.outliers([1000, 1000, 1000, 1000, 900]) == [False] * 4 + [True]


def test_all_identical_flags_nothing():
    assert series.outliers([5.0, 5.0, 5.0]) == [False, False, False]


def test_missing_values_are_neither_judged_nor_counted():
    assert series.outliers([None, 1000, 1010, 990, 600]) == [False, False, False, False, True]


def test_aggregate_uses_the_sample_standard_deviation():
    agg = series.aggregate([10.0, 12.0, 14.0])

    assert (agg.mean, agg.stdev, agg.min, agg.max, agg.n) == (12.0, 2.0, 10.0, 14.0, 3)
    assert agg.cv == pytest.approx(2 / 12)


def test_one_value_has_no_spread():
    agg = series.aggregate([7.0])

    assert (agg.mean, agg.stdev, agg.n) == (7.0, 0.0, 1)
    assert series.aggregate([]) is None


def test_a_dropped_run_leaves_every_metric():
    results = [{"tokens_per_sec": v, "max_temp_c": t}
               for v, t in [(1000, 70), (1010, 71), (990, 70), (500, 95)]]

    summary = series.summarise(results)

    assert summary["dropped"] == [3]
    assert summary["metrics"]["tokens_per_sec"]["n"] == 3
    assert summary["metrics"]["max_temp_c"]["max"] == 71
    assert summary["metrics"]["mfu"] is None          # never reported


def test_record_names_every_member_report():
    reports = [{"gpu": {"name": "RTX 4060"}, "workload": {"model": "TinyLlama 1.1B"},
                "results": {"tokens_per_sec": v}} for v in (1000.0, 1010.0)]

    record = series.build_record(reports, ["report_a.json", "report_b.json"])

    assert record["kind"] == "series"
    assert record["gpu"]["name"] == "RTX 4060"
    assert [m["report"] for m in record["members"]] == ["report_a.json", "report_b.json"]
    assert record["metrics"]["tokens_per_sec"]["mean"] == 1005.0


def test_format_shows_the_spread_and_range():
    agg = {"mean": 12345.0, "cv": 0.012, "min": 12100.0, "max": 12500.0, "n": 3}

    assert series.format_aggregate(agg, "{:,.0f}") == "12,345 ± 1.2 %  (12,100 – 12,500)"
    assert series.format_aggregate({**agg, "n": 1}, "{:,.0f}") == "12,345"
