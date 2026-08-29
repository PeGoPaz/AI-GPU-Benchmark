import gc
import time

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

        # Emit the current step and the latest loss value
        loss = state.log_history[-1].get("loss", 0.0) if state.log_history else 0.0
        self.worker.progress_updated.emit(state.global_step, loss)


class TrainerWorker(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, float)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    max_steps_ready = pyqtSignal(int)

    def __init__(self, steps: int = 150, batch_size: int = 4, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.batch_size = batch_size
        self.completed_steps = 0
        self._is_running = True

    def run(self):
        try:
            self.max_steps_ready.emit(self.steps)
            self.status_updated.emit("Loading Tokenizer & Dataset (may take a moment on first run)...")

            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
                torch_dtype=torch.bfloat16 # Uses 16-bit precision for modern RTX cards
            )

            self.status_updated.emit("Applying LoRA Adapters...")

            # 3. Apply LoRA (Low-Rank Adaptation)
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
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
                max_steps=self.steps,
                logging_steps=1,
                learning_rate=2e-4,
                fp16=False,
                bf16=True, # RTX 40-series optimizes bfloat16 incredibly well
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

            self.status_updated.emit(f"Starting LoRA Fine-Tuning for {self.steps} steps...")

            start_time = time.time()

            # 5. Execute Training!
            trainer.train()

            elapsed_time = time.time() - start_time

            # Use the steps we actually completed. After a Stop the loop exits
            # early, so dividing self.steps by the (short) elapsed time would
            # massively overstate throughput.
            steps_done = self.completed_steps or self.steps
            was_aborted = steps_done < self.steps

            summary = {
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
