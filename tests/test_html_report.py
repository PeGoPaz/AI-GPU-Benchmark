"""The HTML report: one self-contained page per run."""
import re

import numpy as np
import pandas as pd

from gpu_bench.utils import report as report_module
from gpu_bench.utils.html_report import build_html, figure_svg


def _df(sample, n=9):
    df = pd.DataFrame([sample(i) for i in range(n)])
    df["time_sec"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


SUMMARY = {
    "model": "Qwen2.5 1.5B", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "precision": "BF16",
    "batch_size": 2, "seq_len": 512, "gradient_checkpointing": True, "warmup_steps": 10,
    "requested_steps": 100, "tokens_per_step": 4096, "params_b": 1.54,
    "total_steps": 100, "aborted": False, "elapsed_time_sec": 20.0, "steps_per_sec": 5.0,
    "measure_start": 102.0, "measure_end": 106.0,
}
DEVICE = {"name": "NVIDIA GeForce RTX 4060", "driver_version": "580.82",
          "cuda_driver_version": "13.0", "power_limit_w": 115.0}


def _page(sample, **overrides):
    df = _df(sample)
    summary = {**SUMMARY, **overrides}
    rep = report_module.build_report(df, summary, DEVICE, loss=([1, 2, 3], [3.0, 2.8, 2.7]))
    rep["files"] = {"csv": "run_x.csv", "plot": "plot_x.png"}
    return build_html(rep, df, summary)


def test_page_leads_with_the_card_and_the_headline_numbers(telemetry_sample):
    page = _page(telemetry_sample)

    assert "<h1>NVIDIA GeForce RTX 4060</h1>" in page
    assert "Qwen2.5 1.5B · BF16 · batch 2 · 512 tokens · 100 measured steps" in page
    assert "20,480" in page                       # tokens/s
    assert "tokens/J" in page


def test_page_needs_nothing_from_the_network(telemetry_sample):
    """It is the file that gets sent around: no scripts, no remote assets."""
    page = _page(telemetry_sample)

    assert "<script" not in page
    assert not re.search(r"""(src|href)=["']https?://""", page)
    assert page.count("<svg") == 2                # telemetry and loss, inline


def test_page_follows_the_readers_dark_mode(telemetry_sample):
    assert "prefers-color-scheme: dark" in _page(telemetry_sample)


def test_text_from_the_run_is_escaped(telemetry_sample):
    page = _page(telemetry_sample, model="<b>Custom</b> & Co")

    assert "&lt;b&gt;Custom&lt;/b&gt; &amp; Co" in page
    assert "<b>Custom</b>" not in page


def test_loss_panel_is_left_out_without_a_loss_curve(telemetry_sample):
    df = _df(telemetry_sample)
    rep = report_module.build_report(df, SUMMARY, DEVICE)

    page = build_html(rep, df, SUMMARY)

    assert "<h2>Loss</h2>" not in page
    assert page.count("<svg") == 1


def test_inline_svg_has_no_xml_prologue():
    from matplotlib.figure import Figure

    fig = Figure()
    fig.add_subplot(111).plot(np.arange(3), np.arange(3))

    svg = figure_svg(fig)
    assert svg.startswith("<svg")
    assert "<?xml" not in svg


def test_step_panel_appears_with_step_stamps(telemetry_sample):
    page = _page(telemetry_sample, measure_start=100.0, step_times=[104.0, 108.0])

    assert "<h2>Steps</h2>" in page
    assert "Energy per Step" in page


def test_no_step_panel_without_stamps(telemetry_sample):
    assert "<h2>Steps</h2>" not in _page(telemetry_sample)
