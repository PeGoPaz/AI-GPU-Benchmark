# Contributing

AI-GPU-Benchmark is a PyQt6 desktop app that hammers an NVIDIA GPU with a LoRA fine-tune and records what the card does while it happens. Contributions are welcome — here's what you need to know.

## Setup

```bash
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Running the app for real needs Python 3.10+, an NVIDIA GPU and drivers with NVML. Working on it doesn't — see below.

## Tests

The dev tools aren't in `requirements.txt`:

```bash
pip install pytest ruff
ruff check .
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

`tests/conftest.py` swaps torch, transformers, peft and datasets for stubs and mocks NVML, so the suite runs on any laptop with no GPU and no multi-gigabyte install. Window tests carry the `ui` marker and skip themselves when Qt can't start headless. CI does the same on every push and PR: ruff (pinned to 0.16.5), a parse check, and pytest on Python 3.10 and 3.13.

## House rules

- All UI is PyQt6. Anything that blocks — NVML polling, training — lives in a QThread and reports back over signals. The window must never freeze.
- Nothing touches disk mid-run. The logger keeps telemetry in memory and writes only when the user presses Export.
- Tests stay hardware-free: mock the GPU rather than requiring one.
- One logical change per PR, ruff and pytest green, and update `README.md` if users would notice the difference.

## What to work on

Four blocks, roughly ordered — later ones build on earlier ones, and each is a sensible chunk of work by itself. Open an issue before starting anything large so we don't duplicate effort.

### 1. Warm-ups

Self-contained, a file or two each. Good first PRs.

- **Fan and PCIe telemetry.** NVML exposes fan RPM/PWM% and PCIe generation × width (Gen4 x16 and so on). Add them to `TelemetryWorker` and the dashboard — the field-value plumbing in `telemetry.py` already handles "this card doesn't support it".
- **Theming.** Colours currently sit in inline `setStyleSheet()` calls scattered across `main_window.py`. Collect them into one stylesheet; a dark/light toggle becomes easy afterwards.
- **`mypy` in CI.** Types are already partly there. A job in the existing workflow would catch signal/slot mismatches early.

### 2. A configurable run

The gateway to most of the rest: model, step count, batch size and precision are all hardcoded today.

- **Settings panel.** Dropdowns and spin boxes for model (TinyLlama, Phi, Mistral…), step count and batch size, feeding `TrainerWorker` instead of the constants in `start_benchmark()`.
- **Warm-up steps.** The first few steps carry model loading and CUDA compilation overhead and skew the throughput average. Run ~10 warm-up steps (configurable) before starting the timer.
- **Precision toggle.** BF16 / FP16 / FP32, later INT8/INT4 via `bitsandbytes` — one more control in the same panel, and a genuinely interesting axis to measure.

### 3. Results worth keeping

- **Run report.** One Markdown or JSON file per run: GPU, settings, final stats, paths to the CSV and PNG. Makes results shareable, and gives the next two features a stable format to read.
- **Multi-run mode.** Three to five runs back to back, averaged, outliers dropped — the only way the numbers really mean anything. Needs the parameterised run from block 2.
- **Historical overlay.** Load a previous run's CSV and draw its curves behind the current ones, to compare against your own baseline or someone else's card.

### 4. Inference benchmark

The big one. Training is half the picture; plenty of people care more about tokens/sec and time-to-first-token. This is a second benchmark mode with its own worker, metrics and summary — build it on the settings panel and report format from blocks 2 and 3.

## Questions

Open an issue or start a discussion. If you're unsure whether something's in scope, ask first — cheaper than building it and having it turned down.
