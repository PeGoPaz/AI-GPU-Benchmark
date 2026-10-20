"""The native module itself. Skipped unless it was built (`pip install ./native`).

Everything else in the suite runs without the extension; these tests are the
ones that need it, and CI runs them in the job that builds it. None of them
need a GPU: the scripted sampler stands in for NVML.
"""
import sys
import threading
import time

import pytest

nvml_sampler = pytest.importorskip("nvml_sampler")
scripted_sampler = nvml_sampler._testing.scripted_sampler


def _run(sampler, seconds=0.05):
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    return sampler.drain()


def test_version_comes_from_cmake():
    assert nvml_sampler.__version__.count(".") == 2


def test_build_info_names_the_toolchain():
    info = nvml_sampler.build_info()

    assert info["version"] == nvml_sampler.__version__
    assert info["cxx_standard"] >= 201703
    assert info["compiler"]


@pytest.mark.skipif(sys.platform == "linux", reason="a Linux runner may well have NVML")
def test_nvml_is_reported_unavailable_here():
    ok, reason = nvml_sampler.available()

    assert ok is False
    assert reason


def test_opening_without_nvml_raises_with_the_reason():
    with pytest.raises(RuntimeError, match="could not load NVML"):
        nvml_sampler.PowerSampler(library_paths=["/nonexistent/libnvidia-ml.so.1"])


def test_drain_returns_numpy_seconds_and_watts():
    import numpy as np

    before = time.time()
    seconds, watts = _run(scripted_sampler([150_000, 250_000], interval_us=1000))
    after = time.time()

    assert seconds.dtype == np.float64 and watts.dtype == np.float32
    assert len(seconds) == len(watts) > 10
    assert set(watts.tolist()) == {150.0, 250.0}
    # The stamps come from C++'s system_clock, which is precise; time.time()
    # on Windows before Python 3.13 ticks in steps of up to 15.6 ms, and can
    # read earlier than a sample taken before it. Its own resolution is the
    # tolerance.
    slack = time.get_clock_info("time").resolution
    assert before - slack <= seconds[0] <= seconds[-1] <= after + slack


def test_drain_hands_each_sample_over_once():
    sampler = scripted_sampler([100_000], interval_us=1000)
    first, _ = _run(sampler)
    second, _ = sampler.drain()

    assert len(first) > 0
    assert len(second) == 0


def test_failed_reads_are_counted_not_returned():
    sampler = scripted_sampler([100_000, None], interval_us=1000)
    _, watts = _run(sampler)
    stats = sampler.stats()

    assert stats["errors"] > 0
    assert len(watts) == stats["reads"] - stats["errors"]
    assert set(watts.tolist()) == {100.0}


def test_context_manager_stops_the_thread():
    with scripted_sampler([100_000], interval_us=1000) as sampler:
        assert sampler.running
        time.sleep(0.01)
    assert not sampler.running


def test_invalid_interval_is_a_value_error():
    with pytest.raises(ValueError):
        scripted_sampler([100_000], interval_us=0)


def _spin(seconds):
    """Pure Python for the whole duration: the GIL is never released."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def _sample_while(main_thread_work, seconds=0.2):
    sampler = scripted_sampler([100_000, 200_000], interval_us=1000)
    sampler.start()
    main_thread_work(seconds)
    sampler.stop()
    return len(sampler.drain()[0]), sampler.stats()["worst_gap_ms"]


def test_sampling_continues_while_python_holds_the_gil():
    """The reason this module is C++. A Python sampling thread gets the GIL
    from a busy loop only once per switch interval (5 ms by default), so it
    falls to a fraction of its idle rate — ~14% in native/bench. The native
    thread never asks for the GIL and keeps its rate.

    Compared against the same sampler with the main thread idle rather than
    against the nominal 1 kHz: a virtualised CI runner's timers can stretch
    a 1 ms sleep to 2 ms or more, which slows both runs alike and says
    nothing about the GIL.
    """
    idle, _ = _sample_while(time.sleep)
    busy, worst_gap_ms = _sample_while(_spin)

    assert busy >= 0.6 * idle, f"{busy} samples with the GIL held vs {idle} idle"
    assert worst_gap_ms < 50


def test_drain_from_another_thread_is_safe():
    """The C++ ring has one consumer; the wrapper serialises Python callers."""
    sampler = scripted_sampler([100_000], interval_us=200)
    sampler.start()
    counts = []

    def consume():
        total = 0
        for _ in range(50):
            total += len(sampler.drain()[0])
            time.sleep(0.001)
        counts.append(total)

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    sampler.stop()
    leftover = len(sampler.drain()[0])

    assert sum(counts) + leftover == sampler.stats()["reads"]


def test_stats_describe_the_run():
    sampler = scripted_sampler([100_000, 100_000, 300_000], interval_us=1000)
    _run(sampler)
    stats = sampler.stats()

    assert set(stats) == {"reads", "errors", "dropped", "changes", "worst_gap_ms"}
    assert 0 < stats["changes"] < stats["reads"]
    assert sampler.description == "scripted"
    assert sampler.interval_us == 1000
