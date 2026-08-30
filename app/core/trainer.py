import gc
import time
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass(frozen=True)
class ModelSpec:
    """A selectable workload target.

    lora_targets is carried per model rather than hardcoded: attention
    projections are not named consistently across architectures, and a wrong
    name makes get_peft_model attach nothing while training happily reports
    success.
    """

    label: str
    model_id: str
    lora_targets: tuple[str, ...]


# Ungated on the Hub on purpose — a gated model would fail the download with
# an auth error rather than a message anyone can act on.
MODELS = (
    ModelSpec("TinyLlama 1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              ("q_proj", "v_proj")),
    ModelSpec("Qwen2.5 1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
              ("q_proj", "v_proj")),
    ModelSpec("Phi-2 2.7B", "microsoft/phi-2",
              ("q_proj", "v_proj")),
)

DEFAULT_MODEL = MODELS[0]


@dataclass(frozen=True)
class PrecisionSpec:
    """A numeric format the run can be executed in.

    dtype_name is resolved against torch at call time rather than stored as a
    dtype object, so the registry stays importable without torch present.
    """

    label: str
    dtype_name: str
    bf16: bool
    fp16: bool


# BF16 needs Ampere or newer; FP16 works further back but has a much narrower
# exponent range; FP32 runs anywhere and roughly doubles both VRAM and time.
PRECISIONS = (
    PrecisionSpec("BF16", "bfloat16", bf16=True, fp16=False),
    PrecisionSpec("FP16", "float16", bf16=False, fp16=True),
    PrecisionSpec("FP32", "float32", bf16=False, fp16=False),
)

DEFAULT_PRECISION = PRECISIONS[0]


class PyQtProgressCallback(TrainerCallback):
    """Custom Hugging Face callback to send progress back to the PyQt UI."""
    def __init__(self, worker):
        self.worker = worker

    def on_step_end(self, args, state, control, **kwargs):
        # Abort training if the user pressed Stop or closed the window
        if not self.worker._is_running:
            control.should_training_stop = True

        # Record how far we actually got, so the summary reflects reality
        # rather than the requested step count after an early stop.
        self.worker.completed_steps = state.global_step

        # Opening of the timed window. Everything before it carries lazy
        # weight loading and CUDA kernel compilation, which drag the average
        # down and have nothing to do with steady-state throughput.
        if (self.worker.warmup_steps
                and state.global_step == self.worker.warmup_steps):
            self.worker.measure_start = time.time()
            self.worker.status_updated.emit(
                f"Warm-up done after {self.worker.warmup_steps} steps — timing "
                f"the next {self.worker.steps}..."
            )

        # Emit the current step and the latest loss value
        loss = state.log_history[-1].get("loss", 0.0) if state.log_history else 0.0
        self.worker.progress_updated.emit(state.global_step, loss)


class TrainerWorker(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, float)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    max_steps_ready = pyqtSignal(int)

    def __init__(self, steps: int = 150, batch_size: int = 4,
                 model: ModelSpec = DEFAULT_MODEL, warmup_steps: int = 10,
                 precision: PrecisionSpec = DEFAULT_PRECISION, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.batch_size = batch_size
        self.model_spec = model
        self.precision = precision
        self.warmup_steps = warmup_steps
        self.completed_steps = 0
        # Stays None until the timed window opens, which is how a run stopped
        # during warm-up is told apart from one that never started timing.
        self.measure_start: float | None = None
        self._is_running = True

    def run(self):
        try:
            # Warm-up steps really are executed, so the progress bar has to
            # span them too or it would sit at 100% for the whole timed part.
            self.max_steps_ready.emit(self.warmup_steps + self.steps)
            self.status_updated.emit("Loading Tokenizer & Dataset (may take a moment on first run)...")

            model_id = self.model_spec.model_id

            # 1. Load Dataset (A small dataset of English quotes)
            data = load_dataset("Abirate/english_quotes")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            def tokenize(batch):
                return tokenizer(batch["quote"], padding="max_length", truncation=True, max_length=128)

            tokenized_data = data["train"].map(tokenize, batched=True)

            self.status_updated.emit("Loading Base Model into VRAM...")

            # 2. Load Model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda:0",
                torch_dtype=getattr(torch, self.precision.dtype_name),
            )

            self.status_updated.emit("Applying LoRA Adapters...")

            # 3. Apply LoRA (Low-Rank Adaptation)
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=list(self.model_spec.lora_targets),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)

            # 4. Set up Hugging Face Trainer
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=4, # Simulates a batch size of 16 (4x4)
                max_steps=self.warmup_steps + self.steps,
                logging_steps=1,
                learning_rate=2e-4,
                fp16=self.precision.fp16,
                bf16=self.precision.bf16,
                report_to="none" # Disables wandb/tensorboard logging
            )

            trainer = Trainer(
                model=model,
                train_dataset=tokenized_data,
                args=training_args,
                data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(d['input_ids']) for d in data]),
                                            'attention_mask': torch.stack([torch.tensor(d['attention_mask']) for d in data]),
                                            'labels': torch.stack([torch.tensor(d['input_ids']) for d in data])},
                callbacks=[PyQtProgressCallback(self)]
            )

            self.status_updated.emit(
                f"Starting LoRA Fine-Tuning of {self.model_spec.label} "
                f"in {self.precision.label} for {self.steps} steps..."
            )

            start_time = time.time()
            if self.warmup_steps == 0:
                self.measure_start = start_time

            # 5. Execute Training!
            trainer.train()

            # Measure only the steps inside the timed window. After a Stop the
            # loop exits early, so dividing the requested count by the (short)
            # elapsed time would massively overstate throughput; and a run
            # halted during warm-up never opened the window at all.
            if self.measure_start is None:
                elapsed_time = 0.0
                steps_done = 0
            else:
                elapsed_time = time.time() - self.measure_start
                steps_done = max(0, self.completed_steps - self.warmup_steps)
            was_aborted = steps_done < self.steps

            summary = {
                # Recorded so the summary says what was benchmarked, not just
                # how fast it went — the numbers mean nothing without it.
                "model": self.model_spec.label,
                "batch_size": self.batch_size,
                "precision": self.precision.label,
                "warmup_steps": self.warmup_steps,
                "total_steps": steps_done,
                "requested_steps": self.steps,
                "aborted": was_aborted,
                "elapsed_time_sec": round(elapsed_time, 2),
                "steps_per_sec": round(steps_done / elapsed_time, 2) if elapsed_time > 0 else 0.0,
            }

            if was_aborted:
                self.status_updated.emit(
                    f"Training stopped early at {steps_done}/{self.steps} steps."
                )
            else:
                self.status_updated.emit("LoRA Fine-Tuning Complete!")
            self.training_finished.emit(summary)

            # --- Instant VRAM Offload ---
            self.status_updated.emit("Offloading model from VRAM...")
            # `tokenizer` is intentionally left bound: the tokenize() closure
            # defined above still references it, and deleting it here would
            # leave that closure with a dead cell (NameError if called again).
            # It holds no VRAM anyway — only the model and trainer do.
            del trainer, model, tokenized_data, data, config, training_args
            gc.collect()
            torch.cuda.empty_cache()
            self.status_updated.emit("VRAM freed.")

        except Exception as e:
            self.error_occurred.emit(f"HF Training Error: {str(e)}")

    def stop(self):
        """Requests an early stop without blocking the caller.

        This is called from the UI thread, so it must NOT wait() — the current
        training step can take seconds and would freeze the window. The
        callback picks up the flag at the next step boundary; use
        wait_for_exit() when you genuinely need to block (e.g. on app close).
        """
        self._is_running = False

    def wait_for_exit(self, timeout_ms: int = 30000) -> bool:
        """Blocks until the worker thread finishes. Only for shutdown paths."""
        self._is_running = False
        return self.wait(timeout_ms)
