"""TrainerWorker: step accounting, early stop, and non-blocking shutdown."""
import ast
import inspect
import textwrap
import types
from unittest import mock

import app.core.trainer as trainer_module
from app.core.trainer import PyQtProgressCallback, TrainerWorker


class _FakeWorker:
    """Minimal stand-in exposing what the callback touches."""
    def __init__(self):
        self._is_running = True
        self.completed_steps = 0
        self.progress_updated = types.SimpleNamespace(emit=lambda *_: None)


def _drive_callback(steps, stop_at=None):
    worker = _FakeWorker()
    cb = PyQtProgressCallback(worker)
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


def run_worker(steps=150, stop_after=None, elapsed=20.0, **worker_kwargs):
    """Executes the real TrainerWorker.run() against a fake HF Trainer.

    Returns what the worker emitted, plus the patched HF entry points under
    "mocks" so tests can assert what was actually asked of them.
    """
    captured = {}
    worker = TrainerWorker(steps=steps, **worker_kwargs)
    for signal in ("status_updated", "max_steps_ready", "progress_updated"):
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
            for step in range(1, steps + 1):
                if stop_after and step >= stop_after:
                    worker._is_running = False      # user pressed Stop
                state.global_step = step
                for cb in self.callbacks:
                    cb.on_step_end(None, state, control)
                if control.should_training_stop:
                    break

    clock = iter([0.0, elapsed])
    mocks = {name: mock.MagicMock() for name in
             ("TrainingArguments", "AutoTokenizer", "AutoModelForCausalLM",
              "load_dataset", "get_peft_model", "LoraConfig")}
    with mock.patch.multiple(
        trainer_module,
        Trainer=FakeTrainer,
        time=types.SimpleNamespace(time=lambda: next(clock)),
        **mocks,
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
