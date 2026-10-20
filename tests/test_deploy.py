"""deploy/: the Grafana dashboard must ask only for metrics the exporter serves.

A renamed metric would otherwise leave a panel quietly empty, and nothing
in CI would notice; this does.
"""
import json
import re
from pathlib import Path

import pytest

from gpu_bench.utils.prometheus import NAMES

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
DASHBOARD = DEPLOY / "grafana" / "dashboards" / "gpu-bench.json"


def _dashboard():
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _exprs():
    return [target["expr"] for panel in _dashboard()["panels"]
            for target in panel.get("targets", [])]


def test_every_query_uses_only_exported_metrics():
    used = {name for expr in _exprs() for name in re.findall(r"gpu_bench_[a-z_]+", expr)}

    assert used, "the dashboard queries nothing"
    assert used <= NAMES, f"not exported: {sorted(used - NAMES)}"


def test_the_dashboard_covers_the_headline_metrics():
    used = " ".join(_exprs())

    for name in ("gpu_bench_training_tokens_per_second", "gpu_bench_power_watts",
                 "gpu_bench_temperature_celsius", "gpu_bench_sm_clock_hertz",
                 "gpu_bench_throttled", "gpu_bench_energy_joules_total"):
        assert name in used


def test_panels_point_at_the_provisioned_datasource():
    datasource = (DEPLOY / "grafana" / "provisioning" / "datasources" /
                  "prometheus.yml").read_text(encoding="utf-8")
    uid = re.search(r"^\s*uid:\s*(\S+)", datasource, re.M).group(1)

    for panel in _dashboard()["panels"]:
        assert panel["datasource"]["uid"] == uid
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == uid


def test_panel_ids_are_unique_and_the_layout_fits_the_grid():
    panels = _dashboard()["panels"]

    assert len({p["id"] for p in panels}) == len(panels)
    for panel in panels:
        grid = panel["gridPos"]
        assert grid["x"] + grid["w"] <= 24


def test_prometheus_scrapes_the_default_metrics_port():
    config = (DEPLOY / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")

    assert "host.docker.internal:9400" in config


# --- the container image ------------------------------------------------------

DOCKERFILE = (DEPLOY.parent / "Dockerfile").read_text(encoding="utf-8")


def _instructions(keyword):
    return [line.split(None, 1)[1] for line in DOCKERFILE.splitlines()
            if line.split(None, 1)[:1] == [keyword]]


def test_the_base_image_is_pinned_to_an_exact_tag():
    (base,) = [line for line in DOCKERFILE.splitlines() if line.startswith("ARG PYTORCH_IMAGE=")]
    tag = base.split(":", 1)[1]

    assert tag != "latest" and any(ch.isdigit() for ch in tag)


def test_the_image_runs_the_installed_command_as_a_non_root_user():
    tomllib = pytest.importorskip("tomllib")
    scripts = tomllib.loads((DEPLOY.parent / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"]["scripts"]

    (entrypoint,) = _instructions("ENTRYPOINT")
    assert json.loads(entrypoint)[:2] == ["gpu-bench", "--headless"]
    assert "gpu-bench" in scripts
    assert _instructions("USER") == ["bench"]


def test_the_image_exposes_the_port_prometheus_scrapes():
    config = (DEPLOY / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")

    assert _instructions("EXPOSE") == ["9400"]
    assert "bench:9400" in config


def test_the_final_stage_carries_no_compiler():
    final_stage = DOCKERFILE.rsplit("\nFROM ", 1)[1]

    assert "g++" not in final_stage and "build-essential" not in final_stage
