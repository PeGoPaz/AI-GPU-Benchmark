# AI-GPU-Benchmark

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

- A live hardware telemetry dashboard (temperature, memory temperature, thermal headroom, VRAM, power, clock speed)
- A "Start Stress Test" button — launches a LoRA fine-tuning run on TinyLlama-1.1B
- An execution log and progress bar tracking the training loop
- A "Stop" button that ends the run at the next step boundary without freezing the window
- On completion, an in-app thermal & power curve graph rendered directly in the window
- Automatic GPU name detection shown in the window title

---

## How It Works

The benchmark runs a 150-step LoRA fine-tune (bfloat16) on a small English quotes dataset using Hugging Face Transformers and PEFT. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, memory temperature, thermal headroom, VRAM, power, utilization, core clock) into the UI. The BenchmarkLogger collects telemetry snapshots in memory during the run and, on completion, hands a DataFrame to the UI which renders a dual-axis matplotlib plot. The model and all GPU allocations are immediately offloaded and freed from VRAM at the end of each run.

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
    main_window.py             — GUI layout, thread wiring, embedded plot canvas, console log
  core/
    telemetry.py               — TelemetryWorker (QThread): polls NVML for hardware stats, emits GPU name on startup
    trainer.py                 — TrainerWorker (QThread): runs LoRA fine-tuning via HF Trainer, frees VRAM on completion
  utils/
    logger.py                  — BenchmarkLogger: collects telemetry in memory, returns DataFrame for in-app plotting
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | PyQt6 |
| GPU monitoring | nvidia-ml-py (PyNVML) |
| ML training | PyTorch, Transformers, PEFT, TRL, datasets |
| Data & plotting | pandas, matplotlib |

---

## Output

After each benchmark, the thermal & power curves and a summary table are displayed inline in the GUI. Nothing is written to disk during a run — the logger keeps telemetry in memory. Pressing **Export Results** writes a timestamped CSV and plot PNG to `logs/`, and can be pressed repeatedly for the same run. The `results/` directory receives HF checkpoint artifacts from the Trainer during training, but the model is offloaded from VRAM once the run completes.
