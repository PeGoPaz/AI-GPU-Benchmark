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

- A live hardware telemetry dashboard (temperature, hotspot, VRAM, power, clock speed)
- A "Start Stress Test" button — launches a LoRA fine-tuning run on TinyLlama-1.1B
- An execution log and progress bar tracking the training loop
- On completion, an in-app thermal & power curve graph rendered directly in the window
- Automatic GPU name detection shown in the window title

---

## How It Works

The benchmark runs a 150-step LoRA fine-tune (bfloat16) on a small English quotes dataset using Hugging Face Transformers and PEFT. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, hotspot/junction temp, VRAM, power, utilization, core clock) into the UI. The BenchmarkLogger collects telemetry snapshots in memory during the run and, on completion, hands a DataFrame to the UI which renders a dual-axis matplotlib plot. The model and all GPU allocations are immediately offloaded and freed from VRAM at the end of each run.

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

After each benchmark, the thermal & power curves are displayed inline in the GUI. No files are written to disk. The `results/` directory receives HF checkpoint artifacts from the Trainer during training, but the model is offloaded from VRAM once the run completes.
