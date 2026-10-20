"""Live telemetry for Prometheus, from a headless run.

A rented GPU box running a long sweep is watched from somewhere else:
Prometheus scrapes this endpoint and Grafana draws it (deploy/ has both,
ready to run). The latest snapshot of every reading is kept in a
lock-guarded store that the telemetry and trainer signals write into, and
an HTTP server on a daemon thread renders it on every GET /metrics.

The text exposition format is a dozen lines of code, so it is written here
rather than adding prometheus_client as a dependency for one endpoint:
HELP and TYPE per family, base units (bytes, hertz, ratios), _total on the
one counter, and label values escaped as the format requires.
"""
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from gpu_bench.core.telemetry import POWER_LIMIT_BITS, THERMAL_LIMIT_BITS

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "gpu_bench_"


@dataclass(frozen=True)
class Family:
    name: str           # without the prefix
    type: str           # gauge or counter
    help: str


FAMILIES: tuple[Family, ...] = (
    Family("run_info", "gauge", "The run being measured, as labels; always 1."),
    Family("temperature_celsius", "gauge", "GPU core and memory temperature."),
    Family("power_watts", "gauge", "Board power, as NVML's 250 ms reading reports it."),
    Family("power_peak_watts", "gauge",
           "Highest instantaneous power since the previous poll (native sampler only)."),
    Family("power_limit_watts", "gauge", "The enforced power limit."),
    Family("energy_joules_total", "counter", "Energy drawn since the driver loaded."),
    Family("utilization_ratio", "gauge", "GPU utilisation, 0 to 1."),
    Family("memory_used_bytes", "gauge", "VRAM in use."),
    Family("memory_total_bytes", "gauge", "VRAM on the card."),
    Family("sm_clock_hertz", "gauge", "Streaming multiprocessor clock."),
    Family("fan_speed_ratio", "gauge", "Fan duty cycle, 0 to 1."),
    Family("throttled", "gauge", "1 while the clock is held back for this reason."),
    Family("training_step", "gauge", "Optimizer steps completed, warm-up included."),
    Family("training_steps_planned", "gauge", "Optimizer steps the run will take."),
    Family("training_tokens_per_second", "gauge", "Current training throughput."),
    Family("training_loss", "gauge", "Training loss at the latest logged step."),
)
NAMES = frozenset(PREFIX + f.name for f in FAMILIES)


def escape_label(value: object) -> str:
    """A label value as the exposition format requires: \\, \" and newlines escaped."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: float) -> str:
    return repr(float(value)) if value == value else "NaN"


class MetricsState:
    """The latest value of every metric, written by Qt slots, read by HTTP threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # family name → {labels tuple → value}
        self._samples: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}

    def _set(self, family: str, value: float | None, **labels: object) -> None:
        if value is None:
            return
        key = tuple(sorted((k, str(v)) for k, v in labels.items()))
        with self._lock:
            self._samples.setdefault(family, {})[key] = float(value)

    def set_run(self, gpu: str | None, model: str, precision: str, version: str) -> None:
        with self._lock:
            self._samples.pop("run_info", None)
        self._set("run_info", 1, gpu=gpu or "unknown", model=model, precision=precision,
                  version=version)

    def update_telemetry(self, snapshot: dict[str, Any]) -> None:
        """One TelemetryWorker snapshot, in base units."""
        self._set("temperature_celsius", snapshot.get("temp_gpu"), sensor="gpu")
        self._set("temperature_celsius", snapshot.get("temp_memory"), sensor="memory")
        self._set("power_watts", snapshot.get("power_w"))
        self._set("power_peak_watts", snapshot.get("power_peak_w"))
        energy = snapshot.get("energy_mj")
        self._set("energy_joules_total", energy / 1000 if energy is not None else None)
        util = snapshot.get("gpu_util_pct")
        self._set("utilization_ratio", util / 100 if util is not None else None)
        for family, key in (("memory_used_bytes", "vram_used_mb"),
                            ("memory_total_bytes", "vram_total_mb")):
            mib = snapshot.get(key)
            self._set(family, mib * 2**20 if mib is not None else None)
        clock = snapshot.get("sm_clock_mhz")
        self._set("sm_clock_hertz", clock * 1e6 if clock is not None else None)
        fan = snapshot.get("fan_speed_pct")
        self._set("fan_speed_ratio", fan / 100 if fan is not None else None)
        mask = snapshot.get("throttle_reasons")
        if mask is not None:
            self._set("throttled", int(bool(int(mask) & POWER_LIMIT_BITS)), reason="power")
            self._set("throttled", int(bool(int(mask) & THERMAL_LIMIT_BITS)), reason="thermal")

    def set_power_limit(self, watts: float | None) -> None:
        self._set("power_limit_watts", watts)

    def update_progress(self, step: int, planned: int | None,
                        tokens_per_second: float | None) -> None:
        self._set("training_step", step)
        self._set("training_steps_planned", planned)
        self._set("training_tokens_per_second", tokens_per_second)

    def update_loss(self, loss: float) -> None:
        self._set("training_loss", loss)

    def render(self) -> str:
        """The current state in the text exposition format."""
        with self._lock:
            samples = {family: dict(values) for family, values in self._samples.items()}
        lines: list[str] = []
        for family in FAMILIES:
            values = samples.get(family.name)
            if not values:
                continue
            name = PREFIX + family.name
            lines += [f"# HELP {name} {family.help}", f"# TYPE {name} {family.type}"]
            lines += [f"{name}{_labels(labels)} {_number(value)}"
                      for labels, value in sorted(values.items())]
        return "\n".join(lines) + "\n"


def _labels(labels: Iterable[tuple[str, str]]) -> str:
    pairs = [f'{key}="{escape_label(value)}"' for key, value in labels]
    return "{" + ",".join(pairs) + "}" if pairs else ""


class MetricsServer:
    """GET /metrics over HTTP on a daemon thread, for as long as the run lasts."""

    def __init__(self, state: MetricsState, port: int, host: str = "127.0.0.1") -> None:
        self.state = state
        handler = type("Handler", (_Handler,), {"state": state})
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="metrics-server", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _Handler(BaseHTTPRequestHandler):
    state: MetricsState

    def do_GET(self) -> None:
        if self.path.split("?")[0] != "/metrics":
            self._reply(404, "text/plain; charset=utf-8", "Metrics are at /metrics\n")
            return
        self._reply(200, CONTENT_TYPE, self.state.render())

    def _reply(self, status: int, content_type: str, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """Silent: a scrape every 15 s would otherwise fill the run's output."""
