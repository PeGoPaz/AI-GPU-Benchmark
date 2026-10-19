import gc
import time
import types
from dataclasses import dataclass
from functools import lru_cache

from PyQt6.QtCore import QThread, pyqtSignal


@lru_cache(maxsize=1)
def _hf() -> types.SimpleNamespace:
    """The ML stack, imported on first use rather than with this module.

    torch alone takes seconds to import, and this module is imported at
    start-up by the window and the CLI — for its model, precision and preset
    registries, not to train. Deferring the import moves that cost into the
    training thread, where it runs while the window is already responsive,
    and lets the registries (and `main.py --list`) work where torch is not
    installed at all.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    return types.SimpleNamespace(
        torch=torch, load_dataset=load_dataset,
        LoraConfig=LoraConfig, get_peft_model=get_peft_model,
        AutoModelForCausalLM=AutoModelForCausalLM, AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig, Trainer=Trainer,
        TrainerCallback=TrainerCallback, TrainingArguments=TrainingArguments,
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
    # Parameter count in billions, as published on the model card. Known up
    # front so memory and throughput can be estimated before any download.
    params_b: float
    # The token embedding plus LM head (counted once when tied). bitsandbytes
    # leaves both unquantized, so at 4 bits they can be a large share of the
    # footprint: Qwen2.5 7B's 152k vocabulary keeps 1.09B parameters at 16 bits.
    embed_params_b: float


# Ungated on the Hub on purpose — a gated model would fail the download with
# an auth error rather than a message anyone can act on. Listed smallest
# first; from 3B up the weights alone outgrow an 8 GB card at 16 bits.
MODELS = (
    ModelSpec("TinyLlama 1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              ("q_proj", "v_proj"), params_b=1.10, embed_params_b=0.13),
    ModelSpec("Qwen2.5 1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
              ("q_proj", "v_proj"), params_b=1.54, embed_params_b=0.23),
    ModelSpec("Phi-2 2.7B", "microsoft/phi-2",
              ("q_proj", "v_proj"), params_b=2.78, embed_params_b=0.26),
    ModelSpec("Qwen2.5 3B", "Qwen/Qwen2.5-3B-Instruct",
              ("q_proj", "v_proj"), params_b=3.09, embed_params_b=0.31),
    # Phi-3 fuses query, key and value into a single projection, so there is
    # no q_proj or v_proj to attach to — the case lora_targets exists for.
    ModelSpec("Phi-3.5 mini 3.8B", "microsoft/Phi-3.5-mini-instruct",
              ("qkv_proj",), params_b=3.82, embed_params_b=0.20),
    ModelSpec("Qwen2.5 7B", "Qwen/Qwen2.5-7B-Instruct",
              ("q_proj", "v_proj"), params_b=7.62, embed_params_b=1.09),
)

DEFAULT_MODEL = MODELS[0]


@dataclass(frozen=True)
class PrecisionSpec:
    """A numeric format the run can be executed in.

    dtype_name is resolved against torch at call time rather than stored as a
    dtype object, so the registry stays importable without torch present.

    quant_bits marks a format where the frozen base weights are stored
    quantized by bitsandbytes. dtype_name is then the compute dtype: the one
    the weights are dequantized to for each matmul, and the one every
    non-quantized layer (embeddings, norms, LM head) is kept in.
    """

    label: str
    dtype_name: str
    bf16: bool
    fp16: bool
    quant_bits: int | None = None


# BF16 needs Ampere or newer; FP16 works further back but has a much narrower
# exponent range; FP32 runs anywhere and roughly doubles both VRAM and time.
# INT8 and INT4 store the base model quantized — a half and a quarter of the
# 16-bit footprint — which is what fits 3B–7B models on an 8 GB card. INT8
# computes in FP16, the dtype LLM.int8 matmuls produce natively; INT4 is the
# QLoRA recipe (NF4, double quantization) computing in BF16.
PRECISIONS = (
    PrecisionSpec("BF16", "bfloat16", bf16=True, fp16=False),
    PrecisionSpec("FP16", "float16", bf16=False, fp16=True),
    PrecisionSpec("FP32", "float32", bf16=False, fp16=False),
    PrecisionSpec("INT8", "float16", bf16=False, fp16=True, quant_bits=8),
    PrecisionSpec("INT4 (NF4)", "bfloat16", bf16=True, fp16=False, quant_bits=4),
)

DEFAULT_PRECISION = PRECISIONS[0]


_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

# What a run needs beyond the weights before it can take a single step: the
# CUDA context, cuBLAS workspaces and the smallest batch's activations. A
# floor, not a forecast — real activation memory scales with batch and
# sequence length on top of it.
VRAM_RESERVE_GIB = 1.0


def estimate_weights_gib(model: ModelSpec, precision: PrecisionSpec) -> float:
    """Approximate VRAM the base model's weights occupy once loaded.

    Quantized formats only shrink the transformer body; embedding and LM head
    stay in the compute dtype. 4-bit adds about 3% for its quantization
    constants, even after double quantization.
    """
    compute_bytes = _DTYPE_BYTES[precision.dtype_name]
    if precision.quant_bits is None:
        total_bytes = model.params_b * 1e9 * compute_bytes
    else:
        body = model.params_b - model.embed_params_b
        overhead = 1.03 if precision.quant_bits == 4 else 1.0
        total_bytes = (body * 1e9 * precision.quant_bits / 8 * overhead
                       + model.embed_params_b * 1e9 * compute_bytes)
    return total_bytes / 2**30


def fits_in_vram(model: ModelSpec, precision: PrecisionSpec,
                 vram_total_mib: float) -> bool:
    """Whether the weights leave at least the reserve free on this card."""
    budget_gib = vram_total_mib / 1024 - VRAM_RESERVE_GIB
    return estimate_weights_gib(model, precision) <= budget_gib


@dataclass(frozen=True)
class WorkloadPreset:
    """A named, known-good combination of the workload settings.

    Sized for an 8 GB card, each one more work than the last. Seven billion
    parameters only fit there at 4 bits, and Stress stays at 512 tokens:
    Qwen's 152k-token vocabulary makes the FP32 logits alone ~0.6 GB per
    1024-token sequence, with their gradient on top.
    """

    label: str
    model: ModelSpec
    precision: PrecisionSpec
    steps: int
    batch_size: int
    seq_len: int
    gradient_checkpointing: bool
    description: str


def _find(registry, label):
    return next(spec for spec in registry if spec.label == label)


PRESETS = (
    WorkloadPreset("Quick", _find(MODELS, "TinyLlama 1.1B"), _find(PRECISIONS, "BF16"),
                   steps=150, batch_size=4, seq_len=128, gradient_checkpointing=False,
                   description="A smoke test: a minute or two on most cards."),
    WorkloadPreset("Standard", _find(MODELS, "Qwen2.5 1.5B"), _find(PRECISIONS, "BF16"),
                   steps=300, batch_size=2, seq_len=512, gradient_checkpointing=True,
                   description="Long enough for clocks and temperatures to settle."),
    WorkloadPreset("Extended", _find(MODELS, "Qwen2.5 3B"), _find(PRECISIONS, "INT4 (NF4)"),
                   steps=200, batch_size=1, seq_len=1024, gradient_checkpointing=True,
                   description="QLoRA on a 3B model at 1024 tokens — a sustained "
                               "thermal soak."),
    WorkloadPreset("Stress", _find(MODELS, "Qwen2.5 7B"), _find(PRECISIONS, "INT4 (NF4)"),
                   steps=250, batch_size=1, seq_len=512, gradient_checkpointing=True,
                   description="The largest model an 8 GB card can train: 7B at "
                               "4 bits. Expect a long run."),
)

DEFAULT_PRESET = _find(PRESETS, "Standard")


def quantization_config(precision: PrecisionSpec):
    """bitsandbytes settings for a quantized precision, None for a plain dtype."""
    if precision.quant_bits is None:
        return None
    hf = _hf()
    if precision.quant_bits == 4:
        return hf.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # Quantizes the quantization constants too: ~0.37 bits/param
            # saved, about 0.3 GB on a 7B model.
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(hf.torch, precision.dtype_name),
        )
    if precision.quant_bits == 8:
        return hf.BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization: {precision.quant_bits} bits")

# Tokens per training sequence. The heaviest knob on the panel: every layer's
# compute grows with it, attention quadratically, and so does activation memory.
SEQ_LENGTHS = (128, 256, 512, 1024, 2048)

DEFAULT_SEQ_LEN = SEQ_LENGTHS[0]

# Micro-batches per optimizer step. Every step therefore trains on
# batch_size × GRADIENT_ACCUMULATION_STEPS × seq_len tokens.
GRADIENT_ACCUMULATION_STEPS = 4


def pack_tokens(sequences: list[list[int]], seq_len: int,
                separator: int | None = None) -> list[list[int]]:
    """Concatenates tokenized texts and cuts them into blocks of exactly seq_len.

    The dataset is short quotes, a few dozen tokens each. Padding every quote
    out to the sequence length would make a 1024-token run over 95% padding:
    the GPU still does the work, but the model learns from almost none of it.
    Packing keeps every position real text. The separator (the EOS token)
    marks where one quote ends and the next begins; the tail too short to
    fill a block is dropped.
    """
    stream: list[int] = []
    for ids in sequences:
        stream.extend(ids)
        if separator is not None:
            stream.append(separator)

    n_blocks = len(stream) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"Dataset holds {len(stream)} tokens, fewer than one "
            f"{seq_len}-token sequence."
        )
    return [stream[i * seq_len:(i + 1) * seq_len] for i in range(n_blocks)]


def collate_packed(rows):
    """Stacks packed rows into a batch. No padding, so the mask is all ones."""
    torch = _hf().torch
    input_ids = torch.tensor([row["input_ids"] for row in rows])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


class ProgressCallbackLogic:
    """What the Hugging Face progress callback does, minus transformers.

    HF's TrainerCallback is mixed in by progress_callback() when a run
    starts: subclassing it here would import transformers with this module.
    Kept separate so the behaviour is testable on its own.
    """
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

        self.worker.progress_updated.emit(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The loss comes from here rather than from log_history in
        # on_step_end: HF logs a step only after its on_step_end callbacks
        # have run, so reading it there always reported the previous step.
        # The closing summary log carries train_loss instead and is skipped.
        if logs and "loss" in logs:
            self.worker.loss_updated.emit(state.global_step, float(logs["loss"]))


def progress_callback(worker):
    """The callback the Trainer gets: our logic first, HF's no-op hooks after.

    TrainerCallback supplies a do-nothing method for every event the Trainer
    fires, which is why the logic is mixed into it rather than handed over
    alone.
    """
    cls = type("PyQtProgressCallback", (ProgressCallbackLogic, _hf().TrainerCallback), {})
    return cls(worker)


class TrainerWorker(QThread):
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    loss_updated = pyqtSignal(int, float)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    max_steps_ready = pyqtSignal(int)

    def __init__(self, steps: int = 150, batch_size: int = 4,
                 model: ModelSpec = DEFAULT_MODEL, warmup_steps: int = 10,
                 precision: PrecisionSpec = DEFAULT_PRECISION,
                 seq_len: int = DEFAULT_SEQ_LEN,
                 gradient_checkpointing: bool = False, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.batch_size = batch_size
        self.model_spec = model
        self.precision = precision
        self.warmup_steps = warmup_steps
        self.seq_len = seq_len
        self.gradient_checkpointing = gradient_checkpointing
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
            self.status_updated.emit("Loading the ML stack...")
            try:
                hf = _hf()
            except ImportError as e:
                self.error_occurred.emit(
                    f"The ML stack is not installed ({e}). "
                    "Install it with: pip install -r requirements.txt")
                return
            torch = hf.torch
            self.status_updated.emit("Loading Tokenizer & Dataset (may take a moment on first run)...")

            model_id = self.model_spec.model_id

            # 1. Load Dataset (A small dataset of English quotes)
            data = hf.load_dataset("Abirate/english_quotes")
            tokenizer = hf.AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            def tokenize(batch):
                return tokenizer(batch["quote"])

            def pack(batch):
                return {"input_ids": pack_tokens(batch["input_ids"], self.seq_len,
                                                 tokenizer.eos_token_id)}

            train_split = data["train"]
            tokenized = train_split.map(tokenize, batched=True,
                                        remove_columns=train_split.column_names)
            # batch_size=None packs the whole split in one pass; per-batch
            # packing would drop a partial block at every batch boundary.
            packed = tokenized.map(pack, batched=True, batch_size=None,
                                   remove_columns=tokenized.column_names)

            self.status_updated.emit("Loading Base Model into VRAM...")

            # 2. Load Model
            # peft's prepare_model_for_kbit_training() is deliberately not
            # called for the quantized formats. Besides freezing the model,
            # which get_peft_model() already does, it upcasts every
            # non-quantized weight to FP32 — and on Qwen2.5 7B the untied
            # 152k-token embedding and LM head alone become 4.4 GB, more than
            # the whole 4-bit body. That would push the 7B run past an 8 GB
            # card. The 16-bit runs keep those layers in BF16/FP16 as well.
            model = hf.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda:0",
                torch_dtype=getattr(torch, self.precision.dtype_name),
                quantization_config=quantization_config(self.precision),
            )

            self.status_updated.emit("Applying LoRA Adapters...")

            # 3. Apply LoRA (Low-Rank Adaptation)
            config = hf.LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=list(self.model_spec.lora_targets),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            # With every base weight frozen, nothing entering a checkpointed
            # block requires grad, and the reentrant implementation then skips
            # the backward pass through it: the adapters inside train on
            # nothing while the loss curve looks perfectly normal. Non-reentrant
            # checkpointing (below) does not have that failure; making the
            # embedding output require grad covers anything that still
            # takes the reentrant path.
            if self.gradient_checkpointing:
                model.enable_input_require_grads()
            model = hf.get_peft_model(model, config)

            # 4. Set up Hugging Face Trainer
            training_args = hf.TrainingArguments(
                output_dir="./results",
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_steps=self.warmup_steps + self.steps,
                logging_steps=1,
                learning_rate=2e-4,
                fp16=self.precision.fp16,
                bf16=self.precision.bf16,
                # Recomputes activations in the backward pass instead of
                # keeping them: far less VRAM for roughly a third more compute.
                gradient_checkpointing=self.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                report_to="none" # Disables wandb/tensorboard logging
            )

            trainer = hf.Trainer(
                model=model,
                train_dataset=packed,
                args=training_args,
                data_collator=collate_packed,
                callbacks=[progress_callback(self)]
            )

            self.status_updated.emit(
                f"Starting LoRA Fine-Tuning of {self.model_spec.label} "
                f"in {self.precision.label} at {self.seq_len} tokens "
                f"for {self.steps} steps..."
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
            measure_end = None
            if self.measure_start is None:
                elapsed_time = 0.0
                steps_done = 0
            else:
                measure_end = time.time()
                elapsed_time = measure_end - self.measure_start
                steps_done = max(0, self.completed_steps - self.warmup_steps)
            was_aborted = steps_done < self.steps

            summary = {
                # Recorded so the summary says what was benchmarked, not just
                # how fast it went — the numbers mean nothing without it.
                "model": self.model_spec.label,
                "model_id": self.model_spec.model_id,
                "batch_size": self.batch_size,
                "precision": self.precision.label,
                "seq_len": self.seq_len,
                "gradient_checkpointing": self.gradient_checkpointing,
                "params_b": self.model_spec.params_b,
                "tokens_per_step": (self.batch_size * GRADIENT_ACCUMULATION_STEPS
                                    * self.seq_len),
                "warmup_steps": self.warmup_steps,
                "total_steps": steps_done,
                "requested_steps": self.steps,
                "aborted": was_aborted,
                "elapsed_time_sec": round(elapsed_time, 2),
                "steps_per_sec": round(steps_done / elapsed_time, 2) if elapsed_time > 0 else 0.0,
                # Wall-clock bounds of the timed window, so telemetry can be
                # cut to exactly the measured steps (energy per token).
                "measure_start": self.measure_start,
                "measure_end": measure_end,
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
            # `tokenizer` is intentionally left bound: the tokenize() and pack()
            # closures defined above still reference it, and deleting it here would
            # leave that closure with a dead cell (NameError if called again).
            # It holds no VRAM anyway — only the model and trainer do.
            del trainer, model, packed, tokenized, train_split, data, config, training_args
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
