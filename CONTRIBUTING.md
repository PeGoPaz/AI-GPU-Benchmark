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

The dev tooling lives in its own file, so you can work on the project
without a 2 GB torch install:

```bash
pip install -r requirements-dev.txt
ruff check .
mypy
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

`tests/conftest.py` swaps torch, transformers, peft and datasets for stubs and mocks NVML, so the suite runs on any laptop with no GPU and no multi-gigabyte install. Window tests carry the `ui` marker and skip themselves when Qt can't start headless. CI does the same on every push and PR: ruff, mypy, a parse check, and pytest on Python 3.10 and 3.13. Rules live in `ruff.toml` and `mypy.ini`; both tools are pinned in `requirements-dev.txt`, because their default rule sets shift between releases and an unpinned upgrade could turn CI red without a line of project code changing.

## House rules

- All UI is PyQt6. Anything that blocks — NVML polling, training — lives in a QThread and reports back over signals. The window must never freeze.
- Nothing touches disk mid-run. The logger keeps telemetry in memory and writes only when the user presses Export.
- Tests stay hardware-free: mock the GPU rather than requiring one.
- One logical change per PR, all three checks green, and update `README.md` if users would notice the difference.

## What to work on

Four blocks, roughly ordered — later ones build on earlier ones, and each is a sensible chunk of work by itself. Open an issue before starting anything large so we don't duplicate effort.

### 1. Warm-ups

Self-contained, a file or two each. Good first PRs.

- **Unify the two plotting paths.** `main_window._render_plots()` draws the tabs and `logger._generate_plot()` draws the exported PNG, and they now render the same two panels with the same series, differing only in figure layout. Every change to a chart has to be made twice, in two modules. Pull the per-panel drawing into shared functions that take an axes and a DataFrame.
- **Dark/light toggle.** `app/ui/style.py` keeps the palette as constants and builds `STYLESHEET` from them, so a second theme is a second palette plus a menu action that re-applies the sheet. The part worth thinking about is where the choice is remembered between runs; `QSettings` is the obvious answer.
- **Finish the annotations.** `mypy.ini` is deliberately loose — it catches misuse, not missing types. Turning on `--disallow-untyped-defs` reports 33 gaps and `--strict` reports 64, nearly all of them a mechanical `-> None` on a Qt slot. Add them a module at a time, tightening the config as each one goes clean.

### 2. A configurable run

The gateway to most of the rest: model, step count, batch size and precision are all hardcoded today.

- **Quantized precision.** The panel offers BF16, FP16 and FP32; INT8 and INT4 via `bitsandbytes` would extend `PRECISIONS` in `trainer.py` with a quantization config rather than a plain dtype, which is why they were left out of the first pass.

### 3. Results worth keeping

- **Run report.** One Markdown or JSON file per run: GPU, settings, final stats, paths to the CSV and PNG. Makes results shareable, and gives the next two features a stable format to read.
- **Multi-run mode.** Three to five runs back to back, averaged, outliers dropped — the only way the numbers really mean anything. Needs the parameterised run from block 2.
- **Historical overlay.** Load a previous run's CSV and draw its curves behind the current ones, to compare against your own baseline or someone else's card.

### 4. Inference benchmark

The big one. Training is half the picture; plenty of people care more about tokens/sec and time-to-first-token. This is a second benchmark mode with its own worker, metrics and summary — build it on the settings panel and report format from blocks 2 and 3.

## Questions

Open an issue or start a discussion. If you're unsure whether something's in scope, ask first — cheaper than building it and having it turned down.
