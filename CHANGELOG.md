# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The release workflow publishes a version's section below as its release notes.

## [Unreleased]

## [0.1.0] - 2026-10-20

The first release: a LoRA fine-tuning benchmark for NVIDIA GPUs, with a desktop dashboard, a headless mode for servers and an optional C++ power sampler.

### Benchmark

- LoRA fine-tuning workload on six ungated models from TinyLlama 1.1B to Qwen2.5 7B, in BF16, FP16, FP32, INT8 and INT4 (QLoRA, NF4), with sequence packing and gradient checkpointing; four presets sized for an 8 GB card, and a VRAM fit estimate that knows quantization only shrinks the transformer body.
- Warm-up steps excluded from timing; throughput as tokens/sec, estimated TFLOPS, and model and hardware FLOPs utilisation (MFU, HFU) against a dense peak computed from the card's architecture, SM count and clock.
- Energy from NVML's cumulative counter over the timed window, tokens per joule, and energy attributed to every training step.
- Step time p50/p99 and jitter, a first-order thermal model fitted to the temperature curve (τ, the steady temperature, whether the run reached it), throttle reasons, the moment throttling began, and clock and throughput drift across the run.

### Native power sampler

- Optional C++17 extension sampling instantaneous board power at 500 Hz on its own thread, outside the GIL, into a lock-free single-producer/single-consumer ring buffer; NVML loaded at run time, so it builds without the CUDA Toolkit and imports anywhere.
- Builds with GCC, Clang and MSVC; tested against a fake NVML library under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer; prebuilt wheels for Linux and Windows.

### Desktop app

- PyQt6 dashboard: arc gauges and trend tiles coloured by threshold, live pyqtgraph charts with a shared time axis and a crosshair, a phase-aware progress bar, headline result cards over the full summary.
- Light, dark and system themes; restored layout; Help → About with every component version.
- File → Compare runs, and View → Run history: every exported run in a sortable, filterable model/view table with a trend chart.

### Headless and operations

- `gpu-bench --headless` with presets and per-setting flags; `--repeat N` with outliers dropped by the modified z-score; `--cooldown` back to the starting temperature between runs; `--sweep` over a grid of settings, skipping what cannot fit, carrying on past out-of-memory failures, and ranking the results by their Pareto front of speed against efficiency.
- Per run: CSV, PNG, a versioned JSON report and a self-contained HTML page; per series and per sweep: a JSON record, and for a sweep an HTML page with the trade-off chart.
- A SQLite run history with schema migrations; a Prometheus endpoint (`--metrics-port`) with a provisioned Grafana dashboard and Compose stack; a CUDA container image, published to the GitHub Container Registry.

### Engineering

- Packaged with `pyproject.toml` and a `gpu-bench` command; mypy in strict mode, ruff, and a pytest suite with Hypothesis property tests and a coverage floor, all without a GPU.
- CI on Linux, Windows and macOS for the Python and C++ sides, the container image and the wheels; releases built and published from version tags.

[Unreleased]: https://github.com/PeGoPaz/AI-GPU-Benchmark/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PeGoPaz/AI-GPU-Benchmark/releases/tag/v0.1.0
