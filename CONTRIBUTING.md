# Contributing to AI-GPU-Benchmark

Thanks for your interest in contributing! This project is a PyQt6 desktop app for stress-testing NVIDIA GPUs under AI training workloads and capturing real-time telemetry.

## Getting started

```bash
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Requirements: Python 3.10+, NVIDIA GPU with CUDA, NVIDIA drivers with NVML support.

## Coding conventions

- PyQt6 for all UI
- QThread for anything blocking (NVML polling, training) — never run long work on the main thread
- Signals/slots for cross-thread communication
- No file I/O during a benchmark run; the logger keeps telemetry in memory and only writes on explicit export

## Pull requests

1. Fork the repo and create a feature branch
2. One logical change per PR
3. Keep the GUI responsive during long operations
4. Update `README.md` if your change affects user-facing behaviour

---

## Wanted features

These are the things I most want to add. Pick whichever matches your interests — feel free to open an issue before you start so we can align on scope.

### High priority

#### Configurable workload
Right now the benchmark hardcodes TinyLlama-1.1B, 150 steps, batch_size=4, and BF16. Add UI controls (dropdowns, spin boxes) to pick:
- Model (TinyLlama, Phi, Mistral, etc.)
- Step count
- Batch size

#### Warmup phase
The first few steps include model loading and CUDA compilation overhead that skews throughput averages. Run N warmup steps (configurable, default ~10) before starting the timer.

#### Multi-run batch mode
Run 3–5 consecutive benchmarks, average the stats, and discard outliers. Standard practice for reliable benchmarking.

#### Inference benchmark
Training is only half the AI lifecycle. Add a second benchmark mode that measures tokens/sec (generation) and Time-To-First-Token using a standard prompt. Many people care more about inference performance than training.

### Medium priority

#### Precision comparison
Let the user toggle between BF16, FP16, and FP32, or quantized (INT8/INT4 via `bitsandbytes`), to see how precision affects throughput, VRAM usage, and power draw.

#### Historical overlay
Load a previous run's exported CSV and overlay its thermal/power curves on the current plot. Useful for comparing a new run against a baseline or another contributor's hardware.

#### PCIe and fan telemetry
Add fan RPM/PWM% and PCIe generation × width (e.g., Gen4 x16) to the telemetry dashboard via NVML. PCIe bandwidth is relevant for large model loading.

#### Export report
Generate a single Markdown or JSON file summarising a run: GPU name, settings, final stats, and paths to the CSV/plot. Useful for posting results or sharing across a community.

### Nice to have

#### GitHub Actions CI
A basic workflow that runs `ruff` (or `flake8`) and checks that all Python files parse on PRs.

#### `mypy` pre-commit hook
Types are already in the codebase. A pre-commit or CI step running `mypy` would catch signal/slot mismatches and other issues early.

#### Theming
A dark/light mode toggle, or at minimum a stylesheet system that doesn't require digging into inline `styleSheet()` calls.

---

## Questions?

Open an issue or reach out via GitHub Discussions. If you're unsure whether something is in scope, ask first — it's easier than doing the work and having it rejected.
