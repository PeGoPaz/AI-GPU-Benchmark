"""A sweep as one self-contained HTML page: the trade-off chart and the table.

The same rules as the per-run page (gpu_bench.utils.html_report), whose
stylesheet it shares: one file, inline SVG, nothing fetched, light or dark
by the reader's preference. Built from the sweep record alone, so a page can
be regenerated from the JSON at any time.
"""
import html
from typing import Any

from matplotlib.figure import Figure

from gpu_bench.core.series import PICK_LABELS
from gpu_bench.utils.html_report import CSS, figure_svg
from gpu_bench.utils.plots import draw_sweep


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _mean(config: dict[str, Any], key: str) -> float | None:
    metric = (config.get("metrics") or {}).get(key)
    return float(metric["mean"]) if metric else None


def _fmt(config: dict[str, Any], key: str, fmt: str, spread: bool = True) -> str:
    metric = (config.get("metrics") or {}).get(key)
    if not metric:
        return "—"
    text = fmt.format(metric["mean"])
    if spread and metric.get("n", 1) > 1 and metric.get("cv") is not None:
        text += f" ± {metric['cv'] * 100:.1f} %"
    return text


def build_sweep_html(record: dict[str, Any]) -> str:
    """The page for one sweep record (see gpu_bench.batch.Sweep.record)."""
    configs = record.get("configurations") or []
    front = set(record.get("front") or [])
    picks = record.get("picks") or {}
    gpu_name = (record.get("gpu") or {}).get("name") or "Unknown GPU"
    axes = ", ".join(f"{key.replace('_', ' ')} {', '.join(map(str, values))}"
                     for key, values in (record.get("axes") or {}).items())

    ranked = [i for i, c in enumerate(configs) if c.get("status") == "ok"
              and _mean(c, "tokens_per_sec") is not None and _mean(c, "tokens_per_joule")]
    chart = ""
    if ranked:
        points = [(_mean(configs[i], "tokens_per_sec") or 0.0,
                   _mean(configs[i], "tokens_per_joule") or 0.0) for i in ranked]
        position = {config_index: n for n, config_index in enumerate(ranked)}
        fig = Figure(figsize=(10, 5))
        draw_sweep(fig.add_subplot(111), points,
                   [position[i] for i in front if i in position],
                   {name: position[i] for name, i in picks.items() if i in position},
                   [configs[i]["label"] for i in ranked])
        fig.tight_layout()
        chart = (f"<section class='panel'><h2>Speed against efficiency</h2>"
                 f"<figure class='chart'>{figure_svg(fig)}</figure></section>")

    named = {i: [PICK_LABELS[name] for name, j in picks.items() if j == i]
             for i in range(len(configs))}
    rows = []
    for i, config in enumerate(configs):
        ok = config.get("status") == "ok"
        mark = " · ".join(["front", *named[i]]) if i in front else ""
        rows.append(
            f"<tr class='{'front' if i in front else ''}'><td>{_e(config['label'])}</td>"
            f"<td>{_e(_fmt(config, 'tokens_per_sec', '{:,.0f}') if ok else config['status'])}</td>"
            f"<td>{_e(_fmt(config, 'tokens_per_joule', '{:.1f}') if ok else '')}</td>"
            f"<td>{_e(_fmt(config, 'mfu', '{:.1%}', spread=False) if ok else '')}</td>"
            f"<td>{_e(_fmt(config, 'max_temp_c', '{:.0f}', spread=False) if ok else '')}</td>"
            f"<td>{_e(mark)}</td></tr>")
    skipped = "".join(f"<tr><td>{_e(s['label'])}</td><td>{_e(s['reason'])}</td></tr>"
                      for s in record.get("skipped") or [])
    skipped_panel = (f"<section class='panel'><h2>Skipped before running</h2>"
                     f"<table>{skipped}</table></section>") if skipped else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(gpu_name)} — sweep — AI GPU Benchmark</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">AI GPU Benchmark · sweep · {_e(record.get('created_at'))}</p>
<h1>{_e(gpu_name)}</h1>
<p class="sub">{_e(axes)} · {len(configs)} configurations, {_e(record.get('repeat', 1))} run(s) each</p>
</header>
{chart}
<section class="panel"><h2>Configurations</h2>
<table><tr><th>Configuration</th><th>Tokens/sec</th><th>Tokens/J</th><th>MFU</th><th>Max °C</th><th>Pareto</th></tr>
{''.join(rows)}</table></section>
{skipped_panel}
<footer>Schema v{_e(record.get('schema_version'))} · {_e(record.get('app_version'))}</footer>
</main>
</body>
</html>
"""
