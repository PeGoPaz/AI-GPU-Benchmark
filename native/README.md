# nvml_sampler

The optional C++ half of [AI-GPU-Benchmark](../README.md): a power sampler that reads the GPU's instantaneous power draw every few milliseconds on its own thread, outside Python's GIL, for the app to collect once per telemetry poll.

The app does not need it. Without the module, telemetry polls power every 250 ms from Python as it always has.

```bash
pip install ./native          # from the repository root
```

Building needs CMake 3.25+ and a C++17 compiler; pip fetches both pybind11 and scikit-build-core itself. No CUDA Toolkit is required.

## Why NVML is loaded at run time

The module neither links against `libnvidia-ml` nor includes `nvml.h`; it opens the library with `dlopen("libnvidia-ml.so.1")` and resolves the six entry points it needs by name — what pynvml does through ctypes. Linking would make the module unbuildable on a machine without the CUDA Toolkit and unimportable on one without an NVIDIA driver, and would leave the loader untestable in CI. The declarations it needs are copied into `include/power/nvml_api.hpp`, with `static_assert`s pinning the `nvmlFieldValue_t` layout field by field, because a mismatch there would not crash — it would silently read garbage.

The tests build a fake `libnvidia-ml` (`tests/fake_nvml.cpp`) and load it by path through the same code, covering a missing library, a failed `nvmlInit`, a missing entry point and the actual readings, all without a GPU.

## How the sampler works

`power::Sampler` owns one thread that reads a `PowerSource` at a fixed cadence (2 ms by default) and pushes `(unix_ns, milliwatts)` into the ring buffer below; Python drains it once per telemetry poll.

- **Which reading.** `NvmlPowerSource` asks for `NVML_FI_DEV_POWER_INSTANT`, the board's power right now. Where the driver does not support that field it falls back to `nvmlDeviceGetPowerUsage` — smoothed on recent drivers, but available wherever power is reported — and says which one it used.
- **Cadence.** Deadlines advance on the steady clock, so the rate does not drift by however long each read takes. A read that overruns whole intervals skips those slots instead of making them up with a burst of back-to-back reads, which would only cluster samples around the stall.
- **Stamps.** Each sample is stamped with the wall clock, so it lines up with the `time.time()` stamps of the Python telemetry rows.
- **Stopping.** The thread sleeps on a condition variable rather than in `sleep_until`, so `stop()` wakes it immediately even with a long interval; the read itself happens outside the lock, so `stop()` never waits on the driver.
- **Honest rate.** The driver refreshes its reading less often than it can be asked. Besides reads, errors and drops, the sampler counts *changes* — reads whose value differed from the previous one — which is the rate at which new information actually arrived, and what the report shows.

## C++ tests

The sampler core is plain C++ and is tested without Python, using GoogleTest (fetched and pinned by CMake):

```bash
cmake -S native -B build -DPOWER_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

The ring buffer between the sampler thread and Python is lock-free — one producer, one consumer, two atomic counters with acquire/release ordering. Its stress test pushes a million items through a 1024-slot buffer from a second thread and checks nothing is lost or reordered; it is meant to be run under ThreadSanitizer, which flags a missing acquire/release pair as a data race even on runs where the values happen to come out right.
