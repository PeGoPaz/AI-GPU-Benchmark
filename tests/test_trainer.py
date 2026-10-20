"""TrainerWorker: step accounting, early stop, and non-blocking shutdown."""
import ast
import inspect
import textwrap
import types
from unittest import mock

import pytest

import gpu_bench.core.trainer as trainer_module
from gpu_bench.core.trainer import ProgressCallbackLogic, TrainerWorker


class _FakeWorker:
    """Minimal stand-in exposing what the callback touches."""
    def __init__(self, warmup_steps=0, steps=150):
        self._is_running = True
        self.completed_steps = 0
        self.warmup_steps = warmup_steps
        self.steps = steps
        self.measure_start = None
        self.progress_updated = types.SimpleNamespace(emit=lambda *_: None)
        self.status_updated = types.SimpleNamespace(emit=lambda *_: None)
        self.losses = []
        self.loss_updated = types.SimpleNamespace(
            emit=lambda step, loss: self.losses.append((step, loss)))


def _drive_callback(steps, stop_at=None, warmup_steps=0):
    worker = _FakeWorker(warmup_steps=warmup_steps)
    cb = ProgressCallbackLogic(worker)
    state = types.SimpleNamespace(global_step=0, log_history=[{"loss": 0.5}])
    control = types.SimpleNamespace(should_training_stop=False)

    for step in range(1, steps + 1):
        if stop_at and step >= stop_at:
            worker._is_running = False
        state.global_step = step
        cb.on_step_end(None, state, control)
        if control.should_training_stop:
            break
    return worker, control


def run_worker(steps=150, stop_after=None, elapsed=20.0, warmup_elapsed=5.0,
               **worker_kwargs):
    """Executes the real TrainerWorker.run() against a fake HF Trainer.

    `elapsed` is the timed window, not the wall clock: with a warm-up the run
    also burns `warmup_elapsed` before timing opens. Warm-up defaults to off
    here so the older tests keep meaning exactly what they did; the warm-up
    tests ask for it explicitly.

    Returns what the worker emitted, plus the patched HF entry points under
    "mocks" so tests can assert what was actually asked of them.
    """
    captured = {}
    worker_kwargs.setdefault("warmup_steps", 0)
    worker = TrainerWorker(steps=steps, **worker_kwargs)
    for signal in ("status_updated", "max_steps_ready", "progress_updated",
                   "loss_updated"):
        setattr(worker, signal, types.SimpleNamespace(emit=lambda *_: None))
    worker.error_occurred = types.SimpleNamespace(
        emit=lambda m: captured.setdefault("error", m))
    worker.training_finished = types.SimpleNamespace(
        emit=lambda s: captured.setdefault("summary", s))

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get("callbacks", [])

        def train(self):
            state = types.SimpleNamespace(global_step=0, log_history=[{"loss": 0.5}])
            control = types.SimpleNamespace(should_training_stop=False)
            # Warm-up steps are really executed, so the fake runs them too.
            for step in range(1, worker.warmup_steps + steps + 1):
                if stop_after and step >= stop_after:
                    worker._is_running = False      # user pressed Stop
                state.global_step = step
                for cb in self.callbacks:
                    cb.on_step_end(None, state, control)
                if control.should_training_stop:
                    break

    # Ticks in the order run() reads them: start, warm-up boundary, finish.
    ticks = iter([0.0, warmup_elapsed, warmup_elapsed + elapsed]
                 if worker.warmup_steps else [0.0, elapsed])
    last = warmup_elapsed + elapsed

    def clock_now():
        nonlocal last
        try:
            return next(ticks)
        except StopIteration:
            return last
    mocks = {name: mock.MagicMock() for name in
             ("TrainingArguments", "AutoTokenizer", "AutoModelForCausalLM",
              "load_dataset", "get_peft_model", "LoraConfig",
              "BitsAndBytesConfig")}
    # The ML stack as run() sees it: conftest's stubs, with the entry points
    # under test swapped for mocks and the Trainer for the fake above.
    hf = types.SimpleNamespace(**{**vars(trainer_module._hf()),
                                  "Trainer": FakeTrainer, **mocks})
    with mock.patch.multiple(
        trainer_module,
        _hf=lambda: hf,
        time=types.SimpleNamespace(time=clock_now),
    ):
        worker.run()
    captured["mocks"] = mocks
    return captured


def test_callback_tracks_completed_steps():
    worker, control = _drive_callback(42)
    assert worker.completed_steps == 42
    assert control.should_training_stop is False


def test_callback_honours_stop_flag():
    worker, control = _drive_callback(150, stop_at=43)
    assert control.should_training_stop is True
    assert worker.completed_steps == 43


def test_completed_run_reports_all_steps():
    summary = run_worker()["summary"]
    assert summary["total_steps"] == 150
    assert summary["aborted"] is False
    assert summary["steps_per_sec"] == 7.5      # 150 / 20s


def test_aborted_run_uses_real_steps_not_requested():
    """Regression: throughput was computed from the requested step count,
    overstating an early-stopped run by ~3.5x (7.5 vs 2.15 steps/sec)."""
    summary = run_worker(stop_after=43)["summary"]

    assert summary["total_steps"] == 43
    assert summary["requested_steps"] == 150
    assert summary["aborted"] is True
    assert summary["steps_per_sec"] == 2.15     # 43 / 20s, not 150 / 20s


def test_run_completes_without_error():
    """Guards the VRAM-offload path: `del tokenizer` used to break the
    tokenize() closure, leaving it with a dead cell."""
    assert "error" not in run_worker()


def test_offload_keeps_names_referenced_by_closures():
    """`del` must not unbind a name a nested function still closes over.

    Deleting `tokenizer` leaves tokenize() with a dead cell — a latent
    NameError that mocked-out tests cannot surface, so assert it on the source.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(TrainerWorker.run)))
    run_def = tree.body[0]

    deleted = {
        target.id
        for node in ast.walk(run_def)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def free_variables(func):
        """Names a nested function reads but does not bind itself."""
        bound = {a.arg for a in func.args.args + func.args.kwonlyargs}
        if isinstance(func, ast.FunctionDef):
            bound |= {
                t.id
                for n in ast.walk(func)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            }
        read = {
            n.id for n in ast.walk(func)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        return read - bound

    captured = set()
    for node in ast.walk(run_def):
        if node is run_def:
            continue
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            captured |= free_variables(node)

    clash = deleted & captured
    assert not clash, f"deleted while still captured by a closure: {sorted(clash)}"


def test_stop_does_not_block_the_caller():
    """stop() runs on the UI thread; wait() there freezes the window."""
    assert ".wait(" not in inspect.getsource(TrainerWorker.stop)


def test_wait_for_exit_blocks_for_shutdown():
    assert ".wait(" in inspect.getsource(TrainerWorker.wait_for_exit)


# --- selectable workload ------------------------------------------------------

def test_selected_model_reaches_the_hf_calls():
    spec = trainer_module.MODELS[1]
    result = run_worker(model=spec, batch_size=8)

    result["mocks"]["AutoModelForCausalLM"].from_pretrained.assert_called_once()
    assert result["mocks"]["AutoModelForCausalLM"].from_pretrained.call_args[0][0] \
        == spec.model_id
    assert result["mocks"]["AutoTokenizer"].from_pretrained.call_args[0][0] \
        == spec.model_id
    assert result["mocks"]["TrainingArguments"].call_args.kwargs[
        "per_device_train_batch_size"] == 8


def test_lora_targets_come_from_the_model_spec():
    """Hardcoding q_proj/v_proj would silently attach nothing on an
    architecture that names its projections differently."""
    spec = trainer_module.MODELS[2]
    result = run_worker(model=spec)

    assert result["mocks"]["LoraConfig"].call_args.kwargs["target_modules"] \
        == list(spec.lora_targets)


def test_summary_says_what_was_benchmarked():
    spec = trainer_module.MODELS[1]
    summary = run_worker(model=spec, batch_size=8)["summary"]

    assert summary["model"] == spec.label
    assert summary["batch_size"] == 8


def test_every_model_declares_lora_targets():
    for spec in trainer_module.MODELS:
        assert spec.lora_targets, spec.label
        assert "/" in spec.model_id, spec.label
        assert spec.params_b > 0, spec.label


def test_models_are_listed_smallest_first():
    sizes = [spec.params_b for spec in trainer_module.MODELS]
    assert sizes == sorted(sizes)


def test_fused_attention_model_targets_the_fused_projection():
    """Phi-3 has no q_proj/v_proj; asking for them would attach no adapters
    at all and train nothing while reporting a perfectly normal run."""
    phi3 = next(s for s in trainer_module.MODELS if "Phi-3" in s.label)

    assert phi3.lora_targets == ("qkv_proj",)


# --- warm-up ------------------------------------------------------------------

def test_warmup_steps_are_excluded_from_throughput():
    """The first steps carry lazy weight loading and CUDA kernel compilation.
    Counting them reports a rate the card never actually sustained."""
    summary = run_worker(steps=100, warmup_steps=10,
                         warmup_elapsed=30.0, elapsed=20.0)["summary"]

    assert summary["warmup_steps"] == 10
    assert summary["total_steps"] == 100        # measured, warm-up excluded
    assert summary["elapsed_time_sec"] == 20.0  # the timed window only
    assert summary["steps_per_sec"] == 5.0      # 100/20, not 110/50 == 2.2
    assert summary["aborted"] is False


def test_warmup_is_added_to_the_step_budget():
    """Warm-up steps are executed, not skipped, so HF must be asked for them."""
    result = run_worker(steps=100, warmup_steps=10)

    assert result["mocks"]["TrainingArguments"].call_args.kwargs["max_steps"] == 110


def test_stopping_during_warmup_reports_no_throughput():
    """The timed window never opened, so there is no rate to report — and
    certainly not one derived from warm-up steps."""
    summary = run_worker(steps=100, warmup_steps=10, stop_after=4)["summary"]

    assert summary["total_steps"] == 0
    assert summary["elapsed_time_sec"] == 0.0
    assert summary["steps_per_sec"] == 0.0
    assert summary["aborted"] is True


def test_callback_opens_the_timed_window_at_the_boundary():
    worker, _ = _drive_callback(20, warmup_steps=10)

    assert worker.measure_start is not None


def test_no_warmup_times_the_whole_run():
    summary = run_worker(steps=150, warmup_steps=0, elapsed=20.0)["summary"]

    assert summary["warmup_steps"] == 0
    assert summary["steps_per_sec"] == 7.5


# --- precision ----------------------------------------------------------------

def test_each_precision_maps_to_matching_dtype_and_flags():
    """fp16 and bf16 are separate TrainingArguments flags; setting both, or
    neither when a half format was asked for, silently changes the workload."""
    for spec in trainer_module.PRECISIONS:
        result = run_worker(precision=spec)
        args = result["mocks"]["TrainingArguments"].call_args.kwargs
        model_kwargs = result["mocks"]["AutoModelForCausalLM"] \
            .from_pretrained.call_args.kwargs

        assert (args["bf16"], args["fp16"]) == (spec.bf16, spec.fp16), spec.label
        assert not (args["bf16"] and args["fp16"]), spec.label
        assert model_kwargs["torch_dtype"] is getattr(
            trainer_module._hf().torch, spec.dtype_name), spec.label


def test_summary_records_the_precision():
    summary = run_worker(precision=trainer_module.PRECISIONS[2])["summary"]

    assert summary["precision"] == "FP32"


def test_fp32_asks_for_neither_half_flag():
    args = run_worker(precision=trainer_module.PRECISIONS[2])[
        "mocks"]["TrainingArguments"].call_args.kwargs

    assert args["bf16"] is False
    assert args["fp16"] is False


# --- sequence length & packing -----------------------------------------------

def test_packing_fills_every_block_exactly():
    blocks = trainer_module.pack_tokens([[1, 2, 3], [4, 5], [6, 7, 8, 9]], 4)

    assert blocks == [[1, 2, 3, 4], [5, 6, 7, 8]]   # tail [9] dropped


def test_packing_separates_texts_with_the_separator():
    blocks = trainer_module.pack_tokens([[1, 2], [3, 4]], 3, separator=0)

    assert blocks == [[1, 2, 0], [3, 4, 0]]


def test_packing_too_little_text_is_an_error_not_an_empty_dataset():
    """An empty dataset would surface deep inside HF as a confusing
    division or index error, long after the actual cause."""
    with pytest.raises(ValueError, match="fewer than one 64-token"):
        trainer_module.pack_tokens([[1, 2, 3]], 64)


def _pack_fn(result):
    """The pack() closure run() handed to datasets.map, for calling directly."""
    data = result["mocks"]["load_dataset"].return_value
    return data["train"].map.return_value.map.call_args[0][0]


def test_selected_sequence_length_drives_packing():
    result = run_worker(seq_len=512)
    result["mocks"]["AutoTokenizer"].from_pretrained.return_value.eos_token_id = 2

    packed = _pack_fn(result)({"input_ids": [[7] * 300] * 4})["input_ids"]

    assert packed and all(len(block) == 512 for block in packed)


def test_whole_split_is_packed_in_one_pass():
    """Per-batch packing drops a partial block at every 1000-row boundary."""
    data = run_worker()["mocks"]["load_dataset"].return_value
    pack_call = data["train"].map.return_value.map.call_args

    assert pack_call.kwargs["batch_size"] is None


def test_summary_records_the_sequence_length():
    assert run_worker(seq_len=1024)["summary"]["seq_len"] == 1024


# --- gradient checkpointing ---------------------------------------------------

def test_checkpointing_reaches_training_arguments():
    args = run_worker(gradient_checkpointing=True)[
        "mocks"]["TrainingArguments"].call_args.kwargs

    assert args["gradient_checkpointing"] is True
    assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_checkpointing_makes_frozen_inputs_require_grad():
    """Without it the reentrant path skips backward through every block and
    the LoRA adapters inside receive no gradient at all."""
    base = run_worker(gradient_checkpointing=True)[
        "mocks"]["AutoModelForCausalLM"].from_pretrained.return_value

    base.enable_input_require_grads.assert_called_once()


def test_checkpointing_is_off_by_default():
    result = run_worker()
    base = result["mocks"]["AutoModelForCausalLM"].from_pretrained.return_value

    assert result["mocks"]["TrainingArguments"].call_args.kwargs[
        "gradient_checkpointing"] is False
    base.enable_input_require_grads.assert_not_called()
    assert result["summary"]["gradient_checkpointing"] is False


# --- quantized precision ------------------------------------------------------

def _precision(label):
    return next(s for s in trainer_module.PRECISIONS if s.label == label)


def test_plain_dtypes_load_unquantized():
    for label in ("BF16", "FP16", "FP32"):
        result = run_worker(precision=_precision(label))
        load_kwargs = result["mocks"]["AutoModelForCausalLM"] \
            .from_pretrained.call_args.kwargs

        assert load_kwargs["quantization_config"] is None, label
        result["mocks"]["BitsAndBytesConfig"].assert_not_called()


def test_int4_is_the_qlora_recipe():
    result = run_worker(precision=_precision("INT4 (NF4)"))
    bnb = result["mocks"]["BitsAndBytesConfig"]
    load_kwargs = result["mocks"]["AutoModelForCausalLM"] \
        .from_pretrained.call_args.kwargs

    assert bnb.call_args.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": trainer_module._hf().torch.bfloat16,
    }
    assert load_kwargs["quantization_config"] is bnb.return_value


def test_int8_loads_in_8bit_and_trains_in_fp16():
    result = run_worker(precision=_precision("INT8"))
    args = result["mocks"]["TrainingArguments"].call_args.kwargs

    assert result["mocks"]["BitsAndBytesConfig"].call_args.kwargs == {
        "load_in_8bit": True}
    assert (args["fp16"], args["bf16"]) == (True, False)


def test_quantized_runs_skip_the_fp32_upcast():
    """prepare_model_for_kbit_training() upcasts embeddings and LM head to
    FP32 — 4.4 GB on Qwen2.5 7B, enough to push the run past an 8 GB card."""
    assert not hasattr(trainer_module, "prepare_model_for_kbit_training")
    assert not hasattr(trainer_module._hf(), "prepare_model_for_kbit_training")


def test_summary_records_the_quantized_precision():
    summary = run_worker(precision=_precision("INT4 (NF4)"))["summary"]

    assert summary["precision"] == "INT4 (NF4)"


def test_unknown_bit_width_is_rejected():
    spec = trainer_module.PrecisionSpec("INT3", "bfloat16", bf16=True,
                                        fp16=False, quant_bits=3)
    with pytest.raises(ValueError, match="3 bits"):
        trainer_module.quantization_config(spec)


# --- VRAM estimate ------------------------------------------------------------

def _model(label_start):
    return next(s for s in trainer_module.MODELS if s.label.startswith(label_start))


def test_weight_estimate_matches_parameter_count():
    """1.1B parameters at two bytes each is a little over 2 GiB."""
    weights = trainer_module.estimate_weights_gib(_model("TinyLlama"), _precision("BF16"))

    assert weights == pytest.approx(1.10e9 * 2 / 2**30)


def test_fewer_bits_means_smaller_weights():
    order = ("INT4 (NF4)", "INT8", "BF16", "FP32")
    for model in trainer_module.MODELS:
        sizes = [trainer_module.estimate_weights_gib(model, _precision(p)) for p in order]
        assert sizes == sorted(sizes), model.label


def test_quantization_leaves_embedding_and_head_at_16_bits():
    """Treating all 7.6B parameters as 4-bit would claim 3.6 GiB; the 1.09B in
    the embedding and LM head stay BF16 and add another 2 GiB on top."""
    weights = trainer_module.estimate_weights_gib(_model("Qwen2.5 7B"),
                                                  _precision("INT4 (NF4)"))

    assert weights > 7.62e9 * 0.5 / 2**30 + 1.5


def test_7b_fits_an_8gb_card_only_quantized_to_4_bits():
    qwen7b = _model("Qwen2.5 7B")

    assert trainer_module.fits_in_vram(qwen7b, _precision("INT4 (NF4)"), 8192)
    assert not trainer_module.fits_in_vram(qwen7b, _precision("INT8"), 8192)
    assert not trainer_module.fits_in_vram(qwen7b, _precision("BF16"), 8192)


def test_fit_check_keeps_the_reserve_free():
    tiny = _model("TinyLlama")
    weights_mib = trainer_module.estimate_weights_gib(tiny, _precision("BF16")) * 1024

    assert not trainer_module.fits_in_vram(tiny, _precision("BF16"), weights_mib + 512)
    assert trainer_module.fits_in_vram(tiny, _precision("BF16"), weights_mib + 1024)


# --- presets ------------------------------------------------------------------

def test_every_preset_fits_an_8gb_card():
    for preset in trainer_module.PRESETS:
        assert trainer_module.fits_in_vram(preset.model, preset.precision, 8192), \
            preset.label


def test_presets_stay_inside_what_the_panel_accepts():
    for preset in trainer_module.PRESETS:
        assert preset.seq_len in trainer_module.SEQ_LENGTHS, preset.label
        assert 10 <= preset.steps <= 5000, preset.label
        assert 1 <= preset.batch_size <= 64, preset.label


def test_each_preset_is_more_work_than_the_last():
    """Parameters × tokens × steps, with checkpointing's recompute on top."""
    def work(p):
        recompute = 1.5 if p.gradient_checkpointing else 1.0
        return p.model.params_b * p.batch_size * p.seq_len * p.steps * recompute

    amounts = [work(p) for p in trainer_module.PRESETS]
    assert amounts == sorted(amounts)
    assert len(set(amounts)) == len(amounts)


def test_default_preset_is_heavier_than_the_old_hardcoded_run():
    """The old fixed run — 150 steps of TinyLlama at 128 tokens — finished
    before clocks and temperatures had settled."""
    preset = trainer_module.DEFAULT_PRESET

    assert preset.label == "Standard"
    assert preset.model.params_b > 1.1
    assert preset.seq_len > 128


# --- throughput inputs --------------------------------------------------------

def test_summary_carries_what_tokens_and_flops_need():
    summary = run_worker(batch_size=2, seq_len=512,
                         model=trainer_module.MODELS[1])["summary"]

    assert summary["tokens_per_step"] == 2 * trainer_module.GRADIENT_ACCUMULATION_STEPS * 512
    assert summary["params_b"] == trainer_module.MODELS[1].params_b


def test_summary_bounds_the_timed_window():
    summary = run_worker(steps=100, warmup_steps=10,
                         warmup_elapsed=30.0, elapsed=20.0)["summary"]

    assert summary["measure_start"] == 30.0
    assert summary["measure_end"] == 50.0
    assert summary["measure_end"] - summary["measure_start"] == summary["elapsed_time_sec"]


def test_window_stays_unbounded_when_warmup_never_finished():
    summary = run_worker(steps=100, warmup_steps=10, stop_after=4)["summary"]

    assert summary["measure_start"] is None
    assert summary["measure_end"] is None


# --- loss reporting -----------------------------------------------------------

def test_loss_is_reported_with_the_step_it_belongs_to():
    """Reading log_history in on_step_end always gave the previous step's
    loss: HF logs a step only after its on_step_end callbacks have run."""
    worker = _FakeWorker()
    cb = ProgressCallbackLogic(worker)
    state = types.SimpleNamespace(global_step=7, log_history=[{"loss": 9.9}])

    cb.on_log(None, state, None, logs={"loss": 2.5, "learning_rate": 2e-4})

    assert worker.losses == [(7, 2.5)]


def test_closing_summary_log_is_not_a_loss_point():
    """The final log carries train_loss — the run average — not a step loss."""
    worker = _FakeWorker()
    cb = ProgressCallbackLogic(worker)
    state = types.SimpleNamespace(global_step=150, log_history=[])

    cb.on_log(None, state, None, logs={"train_loss": 1.8, "train_runtime": 70.0})
    cb.on_log(None, state, None, logs=None)

    assert worker.losses == []


def test_progress_reports_the_step_alone():
    steps = []
    worker = _FakeWorker()
    worker.progress_updated = types.SimpleNamespace(emit=steps.append)
    cb = ProgressCallbackLogic(worker)
    control = types.SimpleNamespace(should_training_stop=False)

    for step in (1, 2, 3):
        cb.on_step_end(None, types.SimpleNamespace(global_step=step, log_history=[]),
                       control)

    assert steps == [1, 2, 3]


# --- lazy ML stack ------------------------------------------------------------

def _python(code):
    """Runs code in a fresh interpreter: conftest's stubs do not reach it."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=root, env=env, timeout=120)


def test_registries_import_without_the_ml_stack():
    """The window and the CLI import this module for its presets; paying
    for torch there cost seconds at every start-up."""
    result = _python(
        "import sys, gpu_bench.core.trainer, gpu_bench.cli\n"
        "print(sorted(m for m in ('torch', 'transformers', 'peft', 'datasets')"
        " if m in sys.modules))")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_list_works_without_the_ml_stack():
    result = _python("import sys; sys.argv = ['main.py', '--list']\n"
                     "import main; sys.exit(main.main())")

    assert result.returncode == 0, result.stderr
    assert "Presets:" in result.stdout


def test_missing_ml_stack_is_reported_with_the_fix():
    errors = []
    worker = TrainerWorker(steps=10, warmup_steps=0)
    for signal in ("status_updated", "max_steps_ready"):
        setattr(worker, signal, types.SimpleNamespace(emit=lambda *_: None))
    worker.error_occurred = types.SimpleNamespace(emit=errors.append)

    def missing():
        raise ImportError("No module named 'torch'")

    with mock.patch.object(trainer_module, "_hf", missing):
        worker.run()

    assert len(errors) == 1
    assert "No module named 'torch'" in errors[0]
    assert "pip install -r requirements.txt" in errors[0]


def test_trainer_callback_runs_our_logic_over_hfs_hooks():
    class HookBase:
        def on_step_end(self, *a, **k):
            raise AssertionError("HF's no-op shadowed our logic")

        def on_train_begin(self, *a, **k):
            return "hf default"

    hf = types.SimpleNamespace(TrainerCallback=HookBase)
    with mock.patch.object(trainer_module, "_hf", lambda: hf):
        cb = trainer_module.progress_callback(_FakeWorker())

    state = types.SimpleNamespace(global_step=3, log_history=[])
    cb.on_step_end(None, state, types.SimpleNamespace(should_training_stop=False))

    assert cb.worker.completed_steps == 3
    assert cb.on_train_begin() == "hf default"
