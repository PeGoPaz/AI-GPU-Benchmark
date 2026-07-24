# AI-GPU-Benchmark

GPU benchmarking stress test tool — measures AI training throughput and monitors real-time GPU telemetry on NVIDIA cards.

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

- A live hardware telemetry dashboard (temperature, VRAM, power, clock speed)
- A "Start Stress Test" button — launches a LoRA fine-tuning run on TinyLlama-1.1B
- An execution log and progress bar tracking the training loop
- On completion, a CSV dump and a high-res PNG graph are saved under `logs/`

---

## How It Works

The benchmark runs a 150-step LoRA fine-tune (bfloat16) on a small English quotes dataset using Hugging Face Transformers and PEFT. While training runs on the GPU, a separate thread polls NVML every 250 ms and feeds live readings (temperature, VRAM, power, utilization, core clock) into the UI. The BenchmarkLogger captures those telemetry snapshots during the run and, on completion, writes:

- `logs/run_<timestamp>.csv` — raw telemetry rows
- `logs/plot_<timestamp>.png` — dual-axis thermal & power curve (300 DPI)

---

## Architecture

```
main.py                        — entry point, launches the PyQt6 app
app/
  ui/
    main_window.py             — GUI layout, thread wiring, console log
  core/
    telemetry.py               — TelemetryWorker (QThread): polls NVML for hardware stats
    trainer.py                 — TrainerWorker (QThread): runs LoRA fine-tuning via HF Trainer
  utils/
    logger.py                  — BenchmarkLogger: collects telemetry, writes CSV + plot
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

After each run `results/` holds HF checkpoint artifacts (adapter weights, optimizer state, scheduler state). The portable metrics live in `logs/` — ready for comparison across GPUs.
