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

- A live hardware telemetry dashboard (temperature, memory temperature, thermal headroom, power, clock speed, VRAM, fan speed, PCIe link) that begins polling as soon as the app starts
- Automatic GPU name detection shown in the window title
- A **Workload** panel to pick the model (TinyLlama 1.1B, Qwen2.5 1.5B, Phi-2 2.7B, Qwen2.5 3B, Phi-3.5 mini 3.8B, Qwen2.5 7B), step count, batch size, precision, sequence length, warm-up steps and gradient checkpointing, locked for the duration of a run. Under it, an estimate of the model's weights against the card's VRAM turns into a warning when the selection will not fit — the 7B model needs INT4 on an 8 GB card
- A "Start Stress Test" button — launches the LoRA fine-tuning run
- An execution log and progress bar tracking the training loop
- A "Stop" button that ends the run at the next step boundary without freezing the window
- On completion, two tabbed graphs rendered directly in the window — **Thermal & Power** and **Utilization, Clock & Fan** — alongside a **Benchmark Summary** table
- An "Export Results" button that writes the finished run to `logs/` as CSV + PNG

---

## How It Works

The benchmark runs a LoRA fine-tune (rank 8) on a small English quotes dataset using Hugging Face Transformers and PEFT. The quotes are tokenized, joined with an EOS token between them and cut into blocks of exactly the selected sequence length (128 to 2048 tokens) — padding each quote out instead would make a 1024-token run over 95% padding, work the GPU does but the model learns nothing from. Sequence length is the heaviest knob on the panel: compute and activation memory grow with it, attention quadratically. Gradient checkpointing trades the other way — activations are recomputed in the backward pass instead of stored, which cuts VRAM sharply for roughly a third more compute per step, and is what lets long sequences fit on an 8 GB card. It runs non-reentrant, and the embedding output is made to require grad: with every base weight frozen, the reentrant implementation otherwise skips the backward pass through each block and the adapters inside never train. Model, step count and per-device batch size come from the Workload panel — 150 steps of TinyLlama 1.1B at batch 4 by default — and 4 gradient accumulation steps sit on top, so the effective batch is four times the one you pick. Each model carries its own LoRA target modules, since attention projections are not named alike across architectures — Phi-3.5 fuses query, key and value into one `qkv_proj`, where the others have separate `q_proj` and `v_proj`. Every model is ungated on the Hub, so no access token is needed. Precision is selectable between BF16 (the default, Ampere and newer), FP16, FP32, INT8 and INT4 — it drives the weight dtype, the Trainer's mixed-precision flags and, for the last two, a `bitsandbytes` quantization config, and is an axis worth benchmarking in its own right. INT4 is the QLoRA recipe (NF4 with double quantization, BF16 compute) and stores the frozen base model in a quarter of its 16-bit size, which is what lets the 3B–7B models train on an 8 GB card. The fit estimate knows this: quantization only shrinks the transformer body, while the embedding and LM head stay at 16 bits — on Qwen2.5 7B, 2 GiB of the 5.2 GiB total. It counts weights plus a 1 GiB floor for the CUDA context and the smallest activations, so it warns rather than blocks the Start button. PEFT's `prepare_model_for_kbit_training()` is deliberately skipped: it upcasts every non-quantized layer to FP32, and on Qwen2.5 7B the embedding and LM head alone would grow to 4.4 GB. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, memory temperature, thermal headroom, VRAM, power, utilization, core clock, fan speed, PCIe link) into the UI. The BenchmarkLogger collects telemetry snapshots in memory during the run and, on completion, hands a DataFrame to the UI, which draws both matplotlib canvases and computes the summary table — averages and peaks for temperature, power and VRAM, plus throughput and efficiency in steps/sec/W. The model and all GPU allocations are immediately offloaded and freed from VRAM at the end of each run.

Throughput is computed from the steps that actually completed, so stopping a run early reports its real steps/sec instead of overstating it.

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

---

## Architecture

```
main.py                        — entry point, launches the PyQt6 app
app/
  ui/
    main_window.py             — GUI layout, thread wiring, tabbed plot canvases, summary table, console log
    style.py                   — palette and the single application stylesheet
  core/
    telemetry.py               — TelemetryWorker (QThread): polls NVML for hardware stats, emits GPU name on startup
    trainer.py                 — TrainerWorker (QThread): runs LoRA fine-tuning via HF Trainer, frees VRAM on completion
  utils/
    logger.py                  — BenchmarkLogger: collects telemetry in memory, returns a DataFrame, exports CSV + PNG
    plots.py                   — chart panels shared by the window's tabs and the exported PNG
tests/                         — pytest suite; the ML stack and NVML are stubbed, so it runs without a GPU
.github/workflows/ci.yml       — CI: ruff lint, plus tests on Python 3.10 and 3.13
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
| Data & plotting | pandas, matplotlib |
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

`requirements-dev.txt` is the same short list CI installs — no torch, no Hugging Face stack. Tests that build the real window are marked `ui` and skip themselves when Qt cannot start headless. CI runs the same checks on every push to `main` and on every pull request: ruff, mypy, a `compileall` parse check, and the test suite on Python 3.10 and 3.13. Both linters are pinned in `requirements-dev.txt` so an upgrade cannot turn CI red on its own.

---

## Output

After each benchmark, both graphs and a summary table are displayed inline in the GUI. Percentages and clock speed sit on separate axes, so a card throttling its clock under load is visible rather than flattened against the utilization line. Nothing is written to disk during a run — the logger keeps telemetry in memory. Pressing **Export Results** writes a timestamped CSV (`logs/run_<timestamp>.csv`) and a two-panel plot PNG (`logs/plot_<timestamp>.png`), and can be pressed repeatedly for the same run. The Hugging Face Trainer creates `results/` as its `output_dir`, though at the default checkpoint interval (every 500 steps) a 150-step run ends before anything is written there. Both directories are gitignored.
