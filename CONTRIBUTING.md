# Contributing

AI-GPU-Benchmark is a PyQt6 desktop app that hammers an NVIDIA GPU with a LoRA fine-tune and records what the card does while it happens. Contributions are welcome — here's what you need to know.

## Setup

```bash
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
python main.py --headless --preset quick     # the same run without a window
```

Running the app for real needs Python 3.10+, an NVIDIA GPU and drivers with NVML. Working on it doesn't — see below.

## Tests

The dev tooling lives in its own file, so you can work on the project
without a 2 GB torch install:

```bash
pip install -r requirements-dev.txt
ruff check .
mypy
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

For the C++ part in `native/`:

```bash
cmake -S native -B build -DPOWER_BUILD_PYTHON=OFF -DPOWER_WARNINGS_AS_ERRORS=ON
cmake --build build && ctest --test-dir build --output-on-failure
cmake -S native -B build-tsan -DPOWER_BUILD_PYTHON=OFF -DPOWER_SANITIZE=thread   # and the same again
clang-format -i $(git ls-files 'native/*.cpp' 'native/*.hpp')
```

`tests/conftest.py` swaps torch, transformers, peft and datasets for stubs and mocks NVML, so the suite runs on any laptop with no GPU and no multi-gigabyte install. Window tests carry the `ui` marker and skip themselves when Qt can't start headless. CI does the same on every push and PR: ruff, mypy, a parse check, and pytest on Python 3.10 and 3.13. Rules live in `ruff.toml` and `mypy.ini`; both tools are pinned in `requirements-dev.txt`, because their default rule sets shift between releases and an unpinned upgrade could turn CI red without a line of project code changing.

## House rules

- All UI is PyQt6. Anything that blocks — NVML polling, training — lives in a QThread and reports back over signals. The window must never freeze.
- Nothing touches disk mid-run. The logger keeps telemetry in memory and writes only when the user presses Export, or when a headless run ends.
- Numbers the summary shows are computed once, in `app/core/metrics.py`, and formatted once, in `app/utils/display.py`. The window, the CLI and the report all read from there — no second copy.
- Tests stay hardware-free: mock the GPU rather than requiring one.
- One logical change per PR, all three checks green, and update `README.md` if users would notice the difference.

## What to work on

Four blocks, roughly ordered — later ones build on earlier ones, and each is a sensible chunk of work by itself. Open an issue before starting anything large so we don't duplicate effort.

### 1. Warm-ups

Self-contained, a file or two each. Good first PRs.

- **Finish the annotations.** `mypy.ini` is deliberately loose — it catches misuse, not missing types. Turning on `--disallow-untyped-defs` reports 71 gaps and `--strict` reports 162, most of them a mechanical `-> None` on a Qt slot or an untyped `__init__` in the window. Add them a module at a time, tightening the config as each one goes clean; `app/core/metrics.py` and `app/utils/report.py` are nearly there already.
- **BF16 fallback.** BF16 and the INT4 recipe both compute in BF16, which needs compute capability 8.0 (Ampere) or newer. `torch.cuda.get_device_capability()` can drive a warning in the fit hint, the same way VRAM does, and suggest FP16 — or an INT4 variant computing in FP16 — on older cards.
- **Longer sequences.** The selector stops at 2048 tokens: the whole context of TinyLlama and Phi-2, but a fraction of Qwen2.5's 32k and Phi-3.5's 128k. Offering 4096 and up needs `ModelSpec` to carry each model's context length, so the selector can grey out what a model never saw.

### 2. Results worth keeping

Everything here reads the JSON run report (`report_<timestamp>.json`), so it works the same for runs from the window and from `--headless`.

- **Multi-run mode.** Three to five runs back to back, averaged, outliers dropped — the only way the numbers really mean anything. `HeadlessRun` in `app/cli.py` already does one run end to end, so a `--repeat N` is mostly orchestration plus a combined report.
- **Run comparison.** Load two or more reports and show them side by side: tokens/sec, est. TFLOPS, tokens/J, time power- and thermal-limited. The report names its CSV, so the curves of a previous run can be drawn behind the current ones as well.
- **A results table from real cards.** Run the four presets headless on a handful of rented cloud GPUs and publish the reports with a table in the README. Numbers from actual hardware are worth more than any feature on this list.

### 3. Native power sampler (C++)

Telemetry is polled from Python every 250 ms. That is fine for temperatures and clocks, but power on modern cards moves in milliseconds, and a 250 ms poll steps right over the transients — `nvmlDeviceGetPowerUsage` is itself a smoothed reading on recent drivers. The energy counter already covers the total; what is missing is the shape: the peak a PSU actually has to survive.

A small C++ extension (pybind11, built with scikit-build-core) running its own thread outside the GIL, reading the instantaneous power field (`NVML_FI_DEV_POWER_INSTANT`) every 1–10 ms into a ring buffer, and handing Python batches of `(timestamp, mW)` once per poll. `TelemetryWorker` keeps everything else; the sampler only adds a high-rate power series and a true peak. It must degrade to the current behaviour when the extension is not built, and the tests must stay hardware-free — the NVML calls belong behind an interface the tests can replace.

### 4. Inference benchmark

The big one. Training is half the picture; plenty of people care more about tokens/sec and time-to-first-token. This is a second benchmark mode with its own worker, metrics and summary — build it on the settings panel, the presets and the report format, and give it a `--mode inference` in the CLI from the start.

A related, smaller step on the training side: **MFU**. Est. TFLOPS divided by the card's peak dense BF16 rate gives model FLOPs utilisation — the number that actually compares across cards. It needs a peak table keyed by GPU name, which is the part worth getting right.

## Questions

Open an issue or start a discussion. If you're unsure whether something's in scope, ask first — cheaper than building it and having it turned down.
