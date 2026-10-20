# syntax=docker/dockerfile:1
#
# gpu-bench, headless, on any machine with an NVIDIA driver and the NVIDIA
# Container Toolkit:
#
#   docker build -t gpu-bench .
#   docker run --rm --gpus all -v "$PWD/logs:/home/bench/logs" gpu-bench --preset quick
#
# Built on PyTorch's own CUDA runtime image, so torch arrives as the CUDA
# build matching the image rather than whatever pip would pick. The native
# power sampler is compiled in a first stage and only its wheel is carried
# over: the final image has no compiler in it.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# --- 1: wheels -------------------------------------------------------------------
FROM ${PYTORCH_IMAGE} AS wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# scikit-build-core fetches CMake and Ninja into the build environment itself.
COPY native/ native/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels ./native

COPY pyproject.toml requirements.txt README.md ./
COPY gpu_bench/ gpu_bench/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# --- 2: runtime ------------------------------------------------------------------
FROM ${PYTORCH_IMAGE}

# What Qt's core libraries load even without a display, as in CI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libegl1 libgl1 libxkbcommon0 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# torch is already in the image and satisfies requirements.txt as it is;
# pip only adds the rest.
COPY --from=wheels /wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# NVML (telemetry, the power sampler) needs the driver's utility libraries
# mounted, not only the compute ones.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg \
    HF_HOME=/home/bench/.cache/huggingface \
    GPU_BENCH_HOME=/home/bench/.local/share/ai-gpu-benchmark

RUN useradd --create-home --uid 1000 bench \
    && mkdir -p /home/bench/logs /home/bench/.cache/huggingface \
        /home/bench/.local/share/ai-gpu-benchmark \
    && chown -R bench:bench /home/bench
USER bench
WORKDIR /home/bench

# Results, and the Hugging Face cache so models are downloaded once.
VOLUME ["/home/bench/logs", "/home/bench/.cache/huggingface"]
# For --metrics-port 9400 --metrics-host 0.0.0.0.
EXPOSE 9400

ENTRYPOINT ["gpu-bench", "--headless", "--output-dir", "/home/bench/logs"]
CMD ["--preset", "quick"]
