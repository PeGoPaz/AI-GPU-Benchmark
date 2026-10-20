# AI-GPU-Benchmark

[![CI](https://github.com/PeGoPaz/AI-GPU-Benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/PeGoPaz/AI-GPU-Benchmark/actions/workflows/ci.yml)

GPU benchmarking tool — measures AI training throughput via LoRA fine-tuning and monitors real-time GPU telemetry on NVIDIA cards.

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (with NVML support)

### Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

Install the app and its dependencies:

```bash
pip install .
```

That puts a `gpu-bench` command on the path. To run straight from the checkout instead, `pip install -r requirements.txt` installs the same dependencies (`pyproject.toml` reads its list from that file), and `python main.py` then does everything `gpu-bench` does.

### Windows

Everything runs on Windows as well, and CI runs the whole test suite and builds the native module there on every push. In PowerShell:

```powershell
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow local scripts for your user once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Install torch **before** the rest. On Windows, a plain `pip install torch` installs the CPU-only build, which cannot train on the GPU; the CUDA build comes from PyTorch's own package index. Pick the CUDA version matching your driver on [pytorch.org/get-started](https://pytorch.org/get-started/locally/) — for example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

bitsandbytes, for the INT8 and INT4 precisions, has shipped Windows wheels since 0.43, so `requirements.txt` covers it. `nvml.dll` comes with the NVIDIA driver; both pynvml and the native module find it in `System32` (or in `NVIDIA Corporation\NVSMI` on older drivers) without any configuration.

The native module needs a C++ compiler: install **Visual Studio 2022 Build Tools** with the *Desktop development with C++* workload, then, from the repository root:

```powershell
pip install .\native
```

---

## Running

Activate the virtual environment (if not already active), then:

```bash
gpu-bench            # or: python -m gpu_bench, or python main.py from a checkout
gpu-bench --version
```

The GUI window opens with:

- A live hardware telemetry dashboard that begins polling as soon as the app starts. Readings with a natural full scale get arc gauges — temperature against 100 °C, power against the card's enforced limit, VRAM against its total, utilization against 100 % — and the rest get tiles: thermal headroom, memory temperature, clock, fan, PCIe link, and why the clock is throttled. The numeric tiles carry a one-minute trend line scaled to that minute's own range, with the range printed beside it so a two-degree drift is visible without being exaggerated. Everything is coloured by threshold (temperature turns amber at 80 °C and red at 87 °C, headroom at 10 and 5 °C, VRAM at 90 and 97 %, power at 95 % of the limit, any throttle reason amber). A card that does not report a reading shows N/A and says why, rather than a zero
- A header with the detected GPU, its VRAM, driver and CUDA version, enforced power limit and PCIe link, whether power comes from the native sampler or 250 ms polls, and the run's state — Idle, Loading, Warm-up, Measuring, Stopping, Done, Stopped or Failed. Without an NVIDIA driver it says so instead of sitting on "detecting"
- A **Workload** panel with four presets (Quick, Standard, Extended, Stress) and the individual fields behind them — editing any field switches to Custom. It picks the model (TinyLlama 1.1B, Qwen2.5 1.5B, Phi-2 2.7B, Qwen2.5 3B, Phi-3.5 mini 3.8B, Qwen2.5 7B), step count, batch size, precision, sequence length, warm-up steps and gradient checkpointing, locked for the duration of a run. Under it, an estimate of the model's weights against the card's VRAM turns into a warning when the selection will not fit — the 7B model needs INT4 on an 8 GB card
- A toolbar and a **Run** menu with Start (**Ctrl+R**), Stop (**Esc**) and Export (**Ctrl+E**), plus the log toggle (**Ctrl+L**) and a theme menu. Each command is one action shared by its toolbar entry, its shortcut and its button, so they can never disagree about whether it is available
- A "Start Stress Test" button — launches the LoRA fine-tuning run
- An execution log and a phase-aware progress bar: it animates while the model loads, then fills the run's plan as two segments — warm-up steps (executed, not timed) in amber, measured steps in green — with the live rate and an ETA measured from the first completed step underneath, since model loading would otherwise make every early estimate wildly pessimistic
- A "Stop" button that ends the run at the next step boundary without freezing the window
- Live charts drawn with pyqtgraph, visible from the moment a run starts and updated at the 4 Hz telemetry arrives at: a **Telemetry** tab with temperature and power over utilization, fan and clock — two plots on one shared time axis, so a clock dip lines up with the temperature behind it, and zoomable and pannable while the run goes on, with a crosshair that follows the mouse through both plots and reads out every series at that moment, in its own colour — and a **Loss** tab. When the run ends they get a final draw from the complete data, and the **Benchmark Summary** appears: headline cards for throughput (tokens/s), estimated compute, efficiency (tokens/J), peak power and maximum temperature, with every other figure in a **Details** table folded underneath. A card the run cannot support — no token count, no energy reading — is left out rather than shown as a dash. The loss curve is not a convergence plot, the run is far too short for that, but a flat one is the quickest sign the adapters are not receiving gradients
- An "Export Results" button that writes the finished run to `logs/` as CSV + PNG + a JSON run report + a self-contained HTML report, and an **Open report** action (**Ctrl+O**) that opens the page in the browser
- Light and dark themes covering the widgets, the stylesheet and the charts. **View → Theme** follows the operating system by default — and switches live when the OS does — or pins Light or Dark; **Ctrl+Shift+D** flips between the two. A y-axis coloured to match its curve keeps that colour in both themes, and the exported PNG always stays light for printing
- **Help → About** listing what the build is made of — Python, Qt and PyQt, the torch / transformers / peft / bitsandbytes / nvidia-ml-py versions, the native module's version and compiler, how power is being sampled, and the GPU and driver — with a **Copy** button that puts it all on the clipboard for a bug report. The app has its own icon (`gpu_bench/ui/assets/icon.svg`)
- **File → Compare runs…** (**Ctrl+Shift+O**): pick several exported `report_*.json` files to see their results side by side — the best value of each row where a direction makes sense (tokens/s, efficiency higher; temperature and time throttled lower) — with their power and temperature curves overlaid from the CSVs the reports name. A report that cannot be read is listed and skipped, not fatal
- Before the first run, the chart and summary areas say what will appear there and how to get it, instead of leaving an empty gap; without an NVIDIA driver, the chart area says so and lists what still works without one (presets, the VRAM estimate, run comparison)
- A resizable layout: telemetry and workload on the left (scrolling when the window is short), charts, summary and execution log on the right, split by draggable dividers. The log can be dragged shut or hidden from **View → Execution log**, and window size, panel splits and the log's visibility are restored on the next launch
- Theme and workload settings remembered between launches through `QSettings`; a value that no longer exists (a removed model, a hand-edited file) falls back to the default instead of stopping the app from starting

### Headless (SSH, cloud GPUs)

The same benchmark runs without a window — for a rented cloud GPU reached over SSH, or for scripting runs across cards:

```bash
gpu-bench --headless --preset stress
gpu-bench --headless --model qwen2.5-3b --precision int4 --seq-len 1024 --steps 200
gpu-bench --list
```

Flags override the preset, and models and precisions can be named by label, Hugging Face id or any unique prefix (`int4`, `phi-3.5-mini`); an ambiguous one such as `qwen2.5` is refused with the candidates listed. Progress prints about twenty times per run, the summary table at the end, and the CSV, PNG and JSON report are always written to `--output-dir` (default `logs/`). Ctrl+C stops at the next step boundary and still writes the results. The exit status is 0 for a complete run, 130 for one stopped with Ctrl+C, and 1 for a failure — including no telemetry at all, since numbers without the card's readings are not worth keeping. Headless mode runs on a `QCoreApplication`, so it needs no display and never loads the widget stack.

---

## How It Works

The benchmark runs a LoRA fine-tune (rank 8) on a small English quotes dataset using Hugging Face Transformers and PEFT. The quotes are tokenized, joined with an EOS token between them and cut into blocks of exactly the selected sequence length (128 to 2048 tokens) — padding each quote out instead would make a 1024-token run over 95% padding, work the GPU does but the model learns nothing from. Sequence length is the heaviest knob on the panel: compute and activation memory grow with it, attention quadratically. Gradient checkpointing trades the other way — activations are recomputed in the backward pass instead of stored, which cuts VRAM sharply for roughly a third more compute per step, and is what lets long sequences fit on an 8 GB card. It runs non-reentrant, and the embedding output is made to require grad: with every base weight frozen, the reentrant implementation otherwise skips the backward pass through each block and the adapters inside never train. Model, step count and per-device batch size come from the Workload panel — the Standard preset by default — and 4 gradient accumulation steps sit on top, so the effective batch is four times the one you pick. Each model carries its own LoRA target modules, since attention projections are not named alike across architectures — Phi-3.5 fuses query, key and value into one `qkv_proj`, where the others have separate `q_proj` and `v_proj`. Every model is ungated on the Hub, so no access token is needed. Precision is selectable between BF16 (the default, Ampere and newer), FP16, FP32, INT8 and INT4 — it drives the weight dtype, the Trainer's mixed-precision flags and, for the last two, a `bitsandbytes` quantization config, and is an axis worth benchmarking in its own right. INT4 is the QLoRA recipe (NF4 with double quantization, BF16 compute) and stores the frozen base model in a quarter of its 16-bit size, which is what lets the 3B–7B models train on an 8 GB card. The fit estimate knows this: quantization only shrinks the transformer body, while the embedding and LM head stay at 16 bits — on Qwen2.5 7B, 2 GiB of the 5.2 GiB total. It counts weights plus a 1 GiB floor for the CUDA context and the smallest activations, so it warns rather than blocks the Start button. PEFT's `prepare_model_for_kbit_training()` is deliberately skipped: it upcasts every non-quantized layer to FP32, and on Qwen2.5 7B the embedding and LM head alone would grow to 4.4 GB. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, memory temperature, thermal headroom, VRAM, power, utilization, core clock, fan speed, PCIe link, cumulative energy, throttle reasons) into the UI. The BenchmarkLogger collects telemetry snapshots in memory during the run and, on completion, hands a DataFrame to the UI, which draws both matplotlib canvases and computes the summary table — averages and peaks for temperature, power and VRAM, plus the throughput and energy figures described below. The model and all GPU allocations are immediately offloaded and freed from VRAM at the end of each run. torch, transformers, peft and datasets are imported only when a run starts, on the training thread: they take seconds to load, and the window, the presets and `gpu-bench --list` need none of them — so the app opens immediately, and the registries work even where the ML stack is not installed.

### Presets

Sized for an 8 GB card and ordered by the amount of work, each one heavier than the last:

| Preset | Model | Precision | Steps | Batch | Sequence | Checkpointing |
|---|---|---|---|---|---|---|
| Quick | TinyLlama 1.1B | BF16 | 150 | 4 | 128 | off |
| **Standard** (default) | Qwen2.5 1.5B | BF16 | 300 | 2 | 512 | on |
| Extended | Qwen2.5 3B | INT4 | 200 | 1 | 1024 | on |
| Stress | Qwen2.5 7B | INT4 | 250 | 1 | 512 | on |

Quick is the old fixed run, kept as a smoke test; it ends before clocks and temperatures have settled, which is why it is no longer the default. Stress stays at 512 tokens because Qwen's 152k-token vocabulary makes the FP32 logits alone about 0.6 GB per 1024-token sequence, with their gradient on top. A test checks that every preset fits an 8 GB card by the same estimate the panel shows.

Throughput is computed from the steps that actually completed, so stopping a run early reports its real steps/sec instead of overstating it.

### Throughput and energy

Steps/sec alone does not compare across runs — a step at batch 1 and 512 tokens is an eighth of the work of one at batch 2 and 2048 — so the summary also reports:

| Metric | How it is computed |
|---|---|
| Tokens/sec | measured steps × batch × 4 (gradient accumulation) × sequence length, over the timed window |
| Compute (est. TFLOPS) | tokens/sec × FLOPs per token, with 4N per token for LoRA on a frozen base (forward 2N, backward through activations 2N, no weight gradients) and 6N with gradient checkpointing's recompute. Attention-score FLOPs are left out, so long sequences read somewhat low |
| Energy, timed | NVML's cumulative energy counter (Volta and newer) across the timed window — integrated on the card, so it includes the millisecond spikes a 250 ms poll misses. Older cards fall back to integrating the power samples; the label says which was used |
| Efficiency (tokens/J) | measured tokens ÷ timed energy. Loading and warm-up are excluded on both sides of the division |
| Peak power, instant / p99 | with the native module only: the highest and 99th-percentile reading of instantaneous power over the timed window. The 250 ms polls report a smoothed value and step over transients; this is what a PSU actually has to survive |
| Transient ratio | instant peak ÷ instant mean — how far the spikes rise above the steady draw |
| Native sampling | samples per second, and how many of them were new readings. NVML refreshes less often than it can be asked, so the second number is the trace's real resolution — reported rather than assumed |
| Energy, instant trace | the trace integrated over the timed window: a cross-check on the card's energy counter |

The timer does not start with the run. Warm-up steps (10 by default) are executed first and excluded from the measurement, because the opening steps carry lazy weight loading and CUDA kernel compilation — real work, but not the steady-state rate the benchmark is trying to report. Stopping during warm-up reports no throughput at all rather than a number derived from the warm-up itself.

### A note on hotspot (junction) temperature

Earlier versions advertised a hotspot reading. NVML does not expose one: `nvmlDeviceGetTemperature` accepts sensor 0 (GPU core) and, since NVML 13, sensor 1 (`NVML_TEMPERATURE_GPU_MAX`) — nothing else — and the old code's attempts to read sensors 2 and 3 raised `NVML_ERROR_INVALID_ARGUMENT` on every poll and always yielded `N/A`. Tools that do display junction temperature read it through NvAPI or raw BAR0 MMIO, neither of which is available here.

Instead the dashboard reports two values that NVML genuinely provides on Ampere and newer:

| Reading | NVML field | Meaning |
|---|---|---|
| Mem Temp | `NVML_FI_DEV_MEMORY_TEMP` | Memory temperature |
| Thermal Headroom | `NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT` | Degrees remaining before the GPU throttles below base clock |

Both display `N/A` on hardware that does not support the field.

Fan speed and PCIe link follow the same rule. Fan duty cycle is reported as the highest across the card's fans — pegged at 100% means it has run out of thermal room — with RPM alongside it where the driver exposes that entry point; passively cooled datacenter cards show `N/A`. The PCIe link is polled rather than read once, because it downshifts at idle and comes back up under load; the maximum the slot supports is shown next to the current value whenever the two differ, so `Gen1 x16 (max Gen4 x16)` reads as an idling card rather than a broken slot.

### Throttle reasons

A falling SM clock on its own does not say whether the card hit its power limit or its temperature limit. NVML does: `nvmlDeviceGetCurrentClocksEventReasons` returns a bitmask of what is holding the clock down, polled with everything else. The dashboard names the active reasons (power cap, power brake, thermal, HW slowdown), the utilization plot shades power-limited and thermal-limited stretches behind the clock curve, and the summary reports the share of the timed window spent in each. Idle, sync-boost and clock-setting bits are ignored — they are the card resting or obeying a setting, not being held back. The bit values are written out from `nvml.h` rather than taken from pynvml, because NVML is renaming the constants (and the getter) while keeping the bits; drivers older than the rename are asked through the old `…ThrottleReasons` name.

---

## Architecture

```
main.py                        — runs the app from a checkout, without installing it
pyproject.toml                 — package metadata and the gpu-bench command; dependencies come from requirements.txt
gpu_bench/
  __init__.py                  — the version, in one place
  __main__.py                  — entry point: the window, or a headless run with --headless
  cli.py                       — headless mode: argument parsing, a run wired to stdout
  ui/
    main_window.py             — GUI layout, thread wiring, summary table, console log
    live_charts.py             — pyqtgraph live charts, fed one snapshot at a time
    style.py                   — light and dark palettes, the QPalette and stylesheet built from them
    readings.py                — what each telemetry tile shows, and the thresholds that colour it (pure, no Qt)
    icons.py                   — toolbar glyphs as embedded SVG, rendered in the theme's colours
    about.py                   — Help → About: component versions, copyable for bug reports
    compare_window.py          — File → Compare runs: reports side by side, curves overlaid
    assets/icon.svg            — the app icon
    widgets/                   — dashboard widgets (header, status chips, arc gauges, metric tiles, sparklines, result cards, phase progress bar, empty states), themed from the palette
  core/
    telemetry.py               — TelemetryWorker (QThread): polls NVML for hardware stats, emits GPU name on startup
    metrics.py                 — summary numbers: tokens/sec, estimated TFLOPS, energy over the timed window
    power_sampler.py           — opens the native sampler when it is built, explains why not when it is not
    trainer.py                 — TrainerWorker (QThread): runs LoRA fine-tuning via HF Trainer, frees VRAM on completion; imports the ML stack only when a run starts
  utils/
    logger.py                  — BenchmarkLogger: collects telemetry in memory, returns a DataFrame, exports CSV + PNG
    plots.py                   — chart panels shared by the window's tabs and the exported PNG
    report.py                  — the versioned JSON run report written on export
    html_report.py             — the same run as one self-contained HTML page
    display.py                 — summary rows and the progress line, shared by window and CLI
native/                        — optional C++ extension (CMake + pybind11): the high-rate power sampler
tests/                         — pytest suite; the ML stack and NVML are stubbed, so it runs without a GPU
.github/workflows/ci.yml       — CI: ruff, mypy, tests on Linux (3.10, 3.13) and Windows; native build on three OSes, sanitizers, clang-format
pytest.ini                     — pytest configuration
ruff.toml                      — lint rule set (pinned explicitly, not Ruff's shifting defaults)
mypy.ini                       — type-check settings
requirements.txt               — runtime dependencies
requirements-dev.txt           — test and lint tooling (no ML stack)
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | PyQt6 |
| GPU monitoring | nvidia-ml-py (PyNVML) |
| ML training | PyTorch, Transformers, PEFT, datasets, bitsandbytes |
| Data & plotting | pandas, pyqtgraph (live), matplotlib (export) |
| Tests & linting | pytest, ruff, mypy |

---

## Development

The test suite replaces torch, transformers, peft, datasets and NVML with stubs, so it runs on any machine — no GPU and no multi-gigabyte install required:

```bash
pip install -r requirements-dev.txt
ruff check .
mypy
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

`requirements-dev.txt` is the same short list CI installs — no torch, no Hugging Face stack. Tests that build the real window are marked `ui` and skip themselves when Qt cannot start headless. CI runs the same checks on every push to `main` and on every pull request: ruff, mypy, a `compileall` parse check, and the test suite on Python 3.10 and 3.13 on Linux, and on Python 3.13 on Windows. The native module gets its own jobs: a warnings-as-errors build with the C++ tests on Linux, macOS and Windows (MSVC), the same tests under AddressSanitizer + UndefinedBehaviorSanitizer and under ThreadSanitizer, a pinned `clang-format` check, and the Python-side tests against the freshly built module. Both linters are pinned in `requirements-dev.txt` so an upgrade cannot turn CI red on its own.

### Native module (optional)

`native/` holds the project's C++ part, a power sampler built as a Python extension with CMake, pybind11 and scikit-build-core. The app never requires it — everything above works without it — and it is built with one command:

```bash
pip install ./native
```

That needs CMake 3.25+ and a C++17 compiler, but no CUDA Toolkit. When the module is importable and NVML is reachable, the telemetry thread starts it next to its own 250 ms polling and drains it on every poll:

- the dashboard's power reading gains the highest instant reading since the previous poll — `Power: 180 W (peak 262 W)`;
- the Thermal & Power chart draws instantaneous power faintly behind the polls, thinned to a min/max envelope so that single-sample spikes survive the drawing;
- the summary gains the transient rows above, the report a `power_sampler` section, and export adds `logs/power_<timestamp>.csv` with every sample.

The window and the headless run both say at start-up whether native sampling is active and, if not, why — module not built, no driver, or a card without a power reading — and in each of those cases telemetry carries on exactly as before. The module's own tests (`tests/test_native_sampler.py`) skip themselves unless it is installed, so the regular suite stays a pure-Python install. How the C++ side works — loading NVML at run time, the lock-free ring buffer, the sampling cadence — is in [native/README.md](native/README.md).

### Why the sampler is C++

A Python thread can poll NVML every millisecond too — as long as nothing else in the process wants the GIL. During a benchmark, plenty does: the window updates its charts four times a second, pandas aggregates the telemetry, and every Python thread waiting for the GIL gets it only when the holder lets go, at most every switch interval (5 ms by default). The native thread never asks for it. `native/bench/sampler_bench.py` measures exactly that difference: both samplers aim for the same 1 kHz cadence and skip missed slots the same way, while the main thread sleeps, spins in pure Python, or does a dashboard's worth of DataFrame and chart work.

Measured on macOS (Apple M3), Python 3.13.7, scripted source, 1000 µs target (1000 Hz), 5 s per scenario.

| Main thread | Sampler | Rate (Hz) | Gap p50 (ms) | Gap p99 (ms) | Gap max (ms) |
|---|---|---|---|---|---|
| idle | native (C++) | 1000 | 1.00 | 1.33 | 2.33 |
| idle | Python thread | 1000 | 1.00 | 1.32 | 1.61 |
| gil-bound | native (C++) | 1000 | 1.00 | 1.01 | 1.52 |
| gil-bound | Python thread | 136 | 7.50 | 8.70 | 8.93 |
| dashboard | native (C++) | 1000 | 1.00 | 1.42 | 1.50 |
| dashboard | Python thread | 541 | 1.89 | 4.41 | 19.59 |

With the main thread idle the two are indistinguishable — which is the point: the Python thread's problem is not Python's speed but the GIL. Spinning in pure Python cuts it to a reading every ~7 ms, and dashboard-style work leaves gaps of up to ~20 ms, long enough to miss a transient entirely; the native thread keeps 1 kHz throughout. These numbers use the scripted source, so they isolate scheduling from driver cost. On a Linux box with an NVIDIA card, `python native/bench/sampler_bench.py --source nvml` repeats the measurement with both samplers reading the real instant-power field.

---

## Output

During a run the live charts take in each telemetry snapshot as it arrives; pyqtgraph swaps the curves' data rather than redrawing the figure, which is what keeps them smooth through a long run. After it, they get a final draw from the logger's complete data — the same data the export uses — and the summary table appears. The exported PNG is drawn by matplotlib from the same series, colours and throttle spans. Percentages and clock speed sit on separate axes, so a card throttling its clock under load is visible rather than flattened against the utilization line. Nothing is written to disk during a run — the logger keeps telemetry in memory. Pressing **Export Results** writes a timestamped CSV (`logs/run_<timestamp>.csv`), a two-panel plot PNG (`logs/plot_<timestamp>.png`) and a run report (`logs/report_<timestamp>.json`), and can be pressed repeatedly for the same run.

The report is what makes a result comparable with anyone else's: the GPU, driver and CUDA version, the enforced power limit, the versions of torch, transformers, peft and bitsandbytes, every workload setting, every summary figure and the loss curve, plus the names of the CSV and PNG it belongs to. It carries a `schema_version`; fields are added rather than renamed, so older reports stay readable. A run that failed has no summary to report, so its telemetry exports as CSV and PNG only.

Next to the JSON, `logs/report_<timestamp>.html` is the same run for people rather than programs: headline cards, the telemetry and loss charts as inline SVG drawn by the same matplotlib code as the PNG, and the results, workload and environment tables. It is one file with no scripts and nothing fetched from the network, so it can be attached to an email or an issue and opens anywhere, offline; it follows the reader's light or dark preference. Headless runs write it too. The Hugging Face Trainer creates `results/` as its `output_dir`, though at the default checkpoint interval (every 500 steps) every preset ends before anything is written there. Both directories are gitignored.
