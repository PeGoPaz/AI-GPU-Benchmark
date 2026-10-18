"""Native vs pure-Python power sampling, while the interpreter is busy.

The case for the C++ sampler, measured rather than argued. Both samplers aim
for the same cadence and skip missed slots the same way; the only difference
is that the Python one needs the GIL to take each reading, and the native one
never does. Each scenario keeps the main thread doing something:

  idle        sleeping — the best case for both
  gil-bound   a pure-Python loop, holding the GIL throughout
  dashboard   building a DataFrame and drawing a figure, over and over —
              roughly what the window does once a second during a run

  python native/bench/sampler_bench.py                    # scripted source, no GPU
  python native/bench/sampler_bench.py --source nvml      # the real card, both via NVML
  python native/bench/sampler_bench.py --markdown         # a table for the README

Needs the module built (pip install ./native) and numpy; the dashboard
scenario needs pandas and matplotlib as well, as the app itself does.
"""
import argparse
import platform
import sys
import threading
import time

import numpy as np

try:
    import nvml_sampler
except ImportError:
    sys.exit("nvml_sampler is not built — run: pip install ./native")


# --- the two samplers ---------------------------------------------------------------

class PythonSampler:
    """What a Python-only sampler would be: a thread on a fixed cadence."""

    def __init__(self, read, interval_s: float):
        self.read = read
        self.interval_s = interval_s
        self.stamps: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self.read()
            self.stamps.append(time.time())
            deadline += self.interval_s
            now = time.perf_counter()
            if deadline <= now:                       # skip missed slots, like the native one
                deadline += ((now - deadline) // self.interval_s + 1) * self.interval_s
            time.sleep(max(0.0, deadline - time.perf_counter()))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


def make_samplers(source: str, interval_us: int):
    if source == "scripted":
        native = nvml_sampler._testing.scripted_sampler([150_000, 250_000],
                                                        interval_us=interval_us)
        return native, PythonSampler(lambda: 200_000, interval_us / 1e6)

    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    field = pynvml.NVML_FI_DEV_POWER_INSTANT
    native = nvml_sampler.PowerSampler(interval_us=interval_us)
    return native, PythonSampler(lambda: pynvml.nvmlDeviceGetFieldValues(handle, [field]),
                                 interval_us / 1e6)


# --- what the main thread does meanwhile -------------------------------------------------

def idle(seconds: float):
    time.sleep(seconds)


def gil_bound(seconds: float):
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def dashboard(seconds: float):
    import pandas as pd
    from matplotlib.figure import Figure

    rng = np.random.default_rng(0)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        df = pd.DataFrame({"t": np.arange(4000), "w": rng.normal(200, 20, 4000)})
        df["rolling"] = df["w"].rolling(25).mean()
        fig = Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.plot(df["t"], df["w"], df["t"], df["rolling"])
        fig.canvas.draw()


SCENARIOS = {"idle": idle, "gil-bound": gil_bound, "dashboard": dashboard}


# --- measuring ----------------------------------------------------------------------------

def gap_stats(stamps) -> dict:
    # The rate comes from the samples' own span, not the scenario's nominal
    # length: the last chunk of work can run past its deadline, and counting
    # those samples against the shorter time would overstate the rate.
    t = np.asarray(stamps)
    gaps = np.diff(t) * 1000
    if len(gaps) == 0:
        return {"rate": 0.0, "p50": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {"rate": len(gaps) / (t[-1] - t[0]), "p50": float(np.percentile(gaps, 50)),
            "p99": float(np.percentile(gaps, 99)), "max": float(gaps.max())}


def run(source: str, interval_us: int, seconds: float):
    results = []
    for name, work in SCENARIOS.items():
        native, python = make_samplers(source, interval_us)
        native.start()
        python.start()
        work(seconds)
        python.stop()
        native.stop()
        native_stamps = native.drain()[0]
        results.append((name, gap_stats(native_stamps), gap_stats(python.stamps)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", choices=("scripted", "nvml"), default="scripted")
    parser.add_argument("--interval-us", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    target_hz = 1e6 / args.interval_us
    results = run(args.source, args.interval_us, args.seconds)
    where = (f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
             f"{args.source} source, {args.interval_us} µs target ({target_hz:.0f} Hz)")

    if args.markdown:
        print(f"Measured on {where}, {args.seconds:g} s per scenario.\n")
        print("| Main thread | Sampler | Rate (Hz) | Gap p50 (ms) | Gap p99 (ms) | Gap max (ms) |")
        print("|---|---|---|---|---|---|")
        for name, native, python in results:
            for label, s in (("native (C++)", native), ("Python thread", python)):
                print(f"| {name} | {label} | {s['rate']:.0f} | {s['p50']:.2f} | "
                      f"{s['p99']:.2f} | {s['max']:.2f} |")
        return 0

    print(where)
    print(f"{'main thread':<11} {'sampler':<8} {'rate Hz':>8} {'p50 ms':>7} {'p99 ms':>7} {'max ms':>7}")
    for name, native, python in results:
        for label, s in (("native", native), ("python", python)):
            print(f"{name:<11} {label:<8} {s['rate']:>8.0f} {s['p50']:>7.2f} "
                  f"{s['p99']:>7.2f} {s['max']:>7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
