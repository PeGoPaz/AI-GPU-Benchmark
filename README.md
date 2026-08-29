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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

Activate the virtual environment (if not already active), then:

```bash
python main.py
```

The GUI window opens with:

- A live hardware telemetry dashboard (temperature, memory temperature, thermal headroom, power, clock speed, VRAM) that begins polling as soon as the app starts
- Automatic GPU name detection shown in the window title
- A "Start Stress Test" button — launches a LoRA fine-tuning run on TinyLlama-1.1B
- An execution log and progress bar tracking the training loop
- A "Stop" button that ends the run at the next step boundary without freezing the window
- On completion, two tabbed graphs rendered directly in the window — **Thermal & Power** and **Utilization & Clock** — alongside a **Benchmark Summary** table
- An "Export Results" button that writes the finished run to `logs/` as CSV + PNG

---

## How It Works

The benchmark runs a 150-step LoRA fine-tune (bfloat16, rank 8, `q_proj`/`v_proj`) on a small English quotes dataset using Hugging Face Transformers and PEFT. The per-device batch size is 4 with 4 gradient accumulation steps, for an effective batch of 16. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, memory temperature, thermal headroom, VRAM, power, utilization, core clock) into the UI. The BenchmarkLogger collects telemetry snapshots in memory during the run and, on completion, hands a DataFrame to the UI, which draws both matplotlib canvases and computes the summary table — averages and peaks for temperature, power and VRAM, plus throughput and efficiency in steps/sec/W. The model and all GPU allocations are immediately offloaded and freed from VRAM at the end of each run.

Throughput is computed from the steps that actually completed, so stopping a run early reports its real steps/sec instead of overstating it.

### A note on hotspot (junction) temperature

Earlier versions advertised a hotspot reading. NVML does not expose one: `nvmlDeviceGetTemperature` accepts a single sensor (`NVML_TEMPERATURE_COUNT == 1`, sensor 0 = GPU core), and the old code's attempts to read sensors 2 and 3 raised `NVML_ERROR_INVALID_ARGUMENT` on every poll and always yielded `N/A`. Tools that do display junction temperature read it through NvAPI or raw BAR0 MMIO, neither of which is available here.

Instead the dashboard reports two values that NVML genuinely provides on Ampere and newer:

| Reading | NVML field | Meaning |
|---|---|---|
| Mem Temp | `NVML_FI_DEV_MEMORY_TEMP` | Memory temperature |
| Thermal Headroom | `NVML_FI_DEV_TEMPERATURE_GPU_MAX_TLIMIT` | Degrees remaining before the GPU throttles below base clock |

Both display `N/A` on hardware that does not support the field.

---

## Architecture

```
main.py                        — entry point, launches the PyQt6 app
app/
  ui/
    main_window.py             — GUI layout, thread wiring, tabbed plot canvases, summary table, console log
  core/
    telemetry.py               — TelemetryWorker (QThread): polls NVML for hardware stats, emits GPU name on startup
    trainer.py                 — TrainerWorker (QThread): runs LoRA fine-tuning via HF Trainer, frees VRAM on completion
  utils/
    logger.py                  — BenchmarkLogger: collects telemetry in memory, returns a DataFrame, exports CSV + PNG
tests/                         — pytest suite; the ML stack and NVML are stubbed, so it runs without a GPU
.github/workflows/ci.yml       — CI: ruff lint, plus tests on Python 3.10 and 3.13
pytest.ini                     — pytest configuration
ruff.toml                      — lint rule set (pinned explicitly, not Ruff's shifting defaults)
requirements.txt               — runtime dependencies
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | PyQt6 |
| GPU monitoring | nvidia-ml-py (PyNVML) |
| ML training | PyTorch, Transformers, PEFT, datasets |
| Data & plotting | pandas, matplotlib |
| Tests & linting | pytest, ruff |

---

## Development

The test suite replaces torch, transformers, peft, datasets and NVML with stubs, so it runs on any machine — no GPU and no multi-gigabyte install required:

```bash
pip install pytest ruff
ruff check .
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

Tests that build the real window are marked `ui` and skip themselves when Qt cannot start headless. CI runs the same checks on every push to `main` and on every pull request: ruff (pinned to 0.16.5), a `compileall` parse check, and the test suite on Python 3.10 and 3.13.

---

## Output

After each benchmark, the thermal/power and utilization/clock curves and a summary table are displayed inline in the GUI. Nothing is written to disk during a run — the logger keeps telemetry in memory. Pressing **Export Results** writes a timestamped CSV (`logs/run_<timestamp>.csv`) and a two-panel plot PNG (`logs/plot_<timestamp>.png`), and can be pressed repeatedly for the same run. The Hugging Face Trainer creates `results/` as its `output_dir`, though at the default checkpoint interval (every 500 steps) a 150-step run ends before anything is written there. Both directories are gitignored.
