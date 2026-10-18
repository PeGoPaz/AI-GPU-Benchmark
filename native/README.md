# nvml_sampler

The optional C++ half of [AI-GPU-Benchmark](../README.md): a power sampler that reads the GPU's instantaneous power draw every few milliseconds on its own thread, outside Python's GIL, for the app to collect once per telemetry poll.

The app does not need it. Without the module, telemetry polls power every 250 ms from Python as it always has.

```bash
pip install ./native          # from the repository root
```

Building needs CMake 3.25+ and a C++17 compiler; pip fetches both pybind11 and scikit-build-core itself. No CUDA Toolkit is required.
