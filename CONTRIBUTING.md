# Contributing

AI-GPU-Benchmark is a PyQt6 desktop app that hammers an NVIDIA GPU with a LoRA fine-tune and records what the card does while it happens. Contributions are welcome — here's what you need to know.

## Setup

```bash
git clone https://github.com/PeGoPaz/AI-GPU-Benchmark.git
cd AI-GPU-Benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
python main.py --headless --preset quick     # the same run without a window
```

Running the app for real needs Python 3.10+, an NVIDIA GPU and drivers with NVML. Working on it doesn't — see below.

## Tests

The dev tooling lives in its own file, so you can work on the project
without a 2 GB torch install:

```bash
pip install -r requirements-dev.txt
ruff check .
mypy
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg pytest
```

For the C++ part in `native/`:

```bash
cmake -S native -B build -DPOWER_BUILD_PYTHON=OFF -DPOWER_WARNINGS_AS_ERRORS=ON
cmake --build build && ctest --test-dir build --output-on-failure
cmake -S native -B build-tsan -DPOWER_BUILD_PYTHON=OFF -DPOWER_SANITIZE=thread   # and the same again
clang-format -i $(git ls-files 'native/*.cpp' 'native/*.hpp')
```

`tests/conftest.py` swaps torch, transformers, peft and datasets for stubs and mocks NVML, so the suite runs on any laptop with no GPU and no multi-gigabyte install. Window tests carry the `ui` marker and skip themselves when Qt can't start headless. `tests/test_properties.py` holds the invariants — energy adding up across a split, packing losing nothing but the tail, every registry label resolving however it is typed — checked by Hypothesis on generated inputs: random locally, a fixed sequence in CI (`HYPOTHESIS_PROFILE=ci`), so a CI failure reproduces on rerun and prints the input that caused it. A new pure function in `gpu_bench/core` should get a property there as well as examples. CI does the same on every push and PR: ruff, mypy, a parse check, and pytest on Python 3.10 and 3.13. Rules live in `ruff.toml` and `mypy.ini`; both tools are pinned in `requirements-dev.txt`, because their default rule sets shift between releases and an unpinned upgrade could turn CI red without a line of project code changing.

## House rules

- All UI is PyQt6. Anything that blocks — NVML polling, training — lives in a QThread and reports back over signals. The window must never freeze.
- Nothing touches disk mid-run. The logger keeps telemetry in memory and writes only when the user presses Export, or when a headless run ends.
- Numbers the summary shows are computed once, in `gpu_bench/core/metrics.py`, and formatted once, in `gpu_bench/utils/display.py`. The window, the CLI and the report all read from there — no second copy.
- Tests stay hardware-free: mock the GPU rather than requiring one.
- One logical change per PR, all three checks green, and update `README.md` if users would notice the difference.

## Releasing

The version lives in `gpu_bench/__init__.py` and nowhere else; between releases it carries a `.dev0` suffix. To release:

1. Set the version (`0.2.0`), and move the *Unreleased* entries in `CHANGELOG.md` under a `## [0.2.0] - <date>` heading.
2. Run the **Release** workflow by hand from the Actions tab: a dry run that builds the app, the native wheels and the image, and publishes nothing.
3. Commit, then `git tag v0.2.0 && git push origin v0.2.0`. The tag must match the version, and the changelog must have its section — the workflow refuses otherwise — and that section becomes the release notes.
4. Bump the version to the next `.dev0`.

## What to work on

Four blocks, roughly ordered — later ones build on earlier ones, and each is a sensible chunk of work by itself. Open an issue before starting anything large so we don't duplicate effort.

### 1. Warm-ups

Self-contained, a file or two each. Good first PRs.

- **BF16 fallback.** BF16 and the INT4 recipe both compute in BF16, which needs compute capability 8.0 (Ampere) or newer. `torch.cuda.get_device_capability()` can drive a warning in the fit hint, the same way VRAM does, and suggest FP16 — or an INT4 variant computing in FP16 — on older cards.
- **Longer sequences.** The selector stops at 2048 tokens: the whole context of TinyLlama and Phi-2, but a fraction of Qwen2.5's 32k and Phi-3.5's 128k. Offering 4096 and up needs `ModelSpec` to carry each model's context length, so the selector can grey out what a model never saw.

### 2. Results worth keeping

Everything here reads the JSON run report (`report_<timestamp>.json`), so it works the same for runs from the window and from `--headless`.

- **A results table from real cards.** Run the four presets headless, with `--repeat 3`, on a handful of rented cloud GPUs and publish the reports with a table in the README. Numbers from actual hardware are worth more than any feature on this list.

### 3. The native module

The C++ power sampler in `native/` is built, tested, wired in and shipped as prebuilt wheels; what is left are the edges.

- **Per-phase power.** Energy per optimizer step is in (`step_energy` in `gpu_bench/core/metrics.py`); one level finer is the phases inside a step. Stamping the forward pass, the backward pass and the optimizer separately — a Trainer callback has no hooks for them, so module hooks or a subclassed `training_step` — would show which of them the transients come from.

### 4. Inference benchmark

The big one. Training is half the picture; plenty of people care more about tokens/sec and time-to-first-token. This is a second benchmark mode with its own worker, metrics and summary — build it on the settings panel, the presets and the report format, and give it a `--mode inference` in the CLI from the start.

## Questions

Open an issue or start a discussion. If you're unsure whether something's in scope, ask first — cheaper than building it and having it turned down.
