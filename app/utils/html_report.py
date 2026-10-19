"""The run as one self-contained HTML page, for people rather than programs.

The JSON report is for tools; this is what gets sent to someone. One file,
no external requests — the charts are inline SVG from the same matplotlib
code as the PNG, the styling is a <style> block — so it opens anywhere,
offline, and still renders the same in a year. It follows the reader's
light or dark preference.
"""
import html
import io

from matplotlib.figure import Figure

from app.utils.display import headline_cards, summary_rows
from app.utils.plots import build_report_figure, draw_loss

_CSS = """
:root {
  --bg: #f6f7f9; --surface: #ffffff; --border: #dde1e6; --text: #1a1d21;
  --muted: #69707a; --accent: #3a6ea5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1f2023; --surface: #2b2c30; --border: #3d3f44; --text: #ececec;
    --muted: #a3a7ad; --accent: #6aa5dc;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 15px/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 48px; }
.eyebrow { color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .08em;
           text-transform: uppercase; margin: 0; }
h1 { font-size: 28px; margin: 4px 0; }
h2 { font-size: 16px; margin: 0 0 12px; }
.sub { color: var(--muted); margin: 0 0 24px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
         gap: 12px; margin-bottom: 24px; }
.card, .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
.card { padding: 14px 16px; }
.card .label { color: var(--muted); font-size: 11px; font-weight: 700;
               letter-spacing: .06em; text-transform: uppercase; }
.card .value { color: var(--accent); font-size: 28px; font-weight: 700; }
.card .unit { color: var(--muted); font-size: 13px; margin-left: 4px; }
.card .caption { color: var(--muted); font-size: 12px; }
.panel { padding: 16px; margin-bottom: 24px; }
/* The charts are drawn on white; keep them on white in dark mode too. */
.chart { background: #ffffff; border-radius: 6px; padding: 8px; margin: 0; }
.chart svg { width: 100%; height: auto; display: block; }
.tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
td { padding: 6px 4px; border-bottom: 1px solid var(--border); vertical-align: top; }
td:first-child { color: var(--muted); padding-right: 16px; }
footer { color: var(--muted); font-size: 12px; margin-top: 8px; }
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def figure_svg(fig: Figure) -> str:
    """A figure as an inline <svg> element, without the XML prologue."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    text = buffer.getvalue()
    return text[text.index("<svg"):]


def _table(rows) -> str:
    body = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _loss_svg(loss) -> str | None:
    steps, values = loss
    if not steps:
        return None
    fig = Figure(figsize=(10, 3))
    draw_loss(fig.add_subplot(111), list(steps), list(values))
    fig.tight_layout()
    return figure_svg(fig)


def build_html(report: dict, df, summary: dict, trace=None) -> str:
    """The page for one run: headline cards, charts, and every number behind them."""
    gpu = report.get("gpu") or {}
    workload = report.get("workload") or {}
    gpu_name = gpu.get("name") or "Unknown GPU"
    model = workload.get("model") or "unknown model"

    subtitle = " · ".join(str(part) for part in (
        model, workload.get("precision"),
        f"batch {workload['batch_size']}" if workload.get("batch_size") else None,
        f"{workload['seq_len']} tokens" if workload.get("seq_len") else None,
        f"{summary.get('total_steps')} measured steps",
    ) if part)

    cards = "".join(
        f"<div class='card'><div class='label'>{_e(c.label)}</div>"
        f"<div><span class='value'>{_e(c.value)}</span><span class='unit'>{_e(c.unit)}</span></div>"
        f"<div class='caption'>{_e(c.caption)}</div></div>"
        for c in headline_cards(df, summary, trace))

    telemetry_svg = figure_svg(build_report_figure(df, trace))
    loss = report.get("loss") or {}
    loss_svg = _loss_svg((loss.get("steps", []), loss.get("values", [])))
    loss_panel = (f"<section class='panel'><h2>Loss</h2><figure class='chart'>{loss_svg}"
                  f"</figure></section>") if loss_svg else ""

    sampler = report.get("power_sampler") or {}
    environment = [
        ("GPU", gpu_name),
        ("Driver / CUDA", f"{gpu.get('driver_version') or '?'} / "
                          f"{gpu.get('cuda_driver_version') or '?'}"),
        ("Power limit", f"{gpu['power_limit_w']:.0f} W" if gpu.get("power_limit_w") else "?"),
        ("Power sampling", ("native, " if sampler.get("active") else "")
         + str(sampler.get("detail", "?"))),
        ("OS / Python", f"{(report.get('host') or {}).get('os', '?')} / "
                        f"{(report.get('host') or {}).get('python', '?')}"),
    ] + [(name, version or "not installed")
         for name, version in (report.get("packages") or {}).items()]
    workload_rows = [(k.replace("_", " "), v) for k, v in workload.items() if v is not None]
    files = ", ".join(_e(v) for v in (report.get("files") or {}).values())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(gpu_name)} — {_e(model)} — AI GPU Benchmark</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">AI GPU Benchmark · {_e(report.get('created_at'))}</p>
<h1>{_e(gpu_name)}</h1>
<p class="sub">{_e(subtitle)}</p>
</header>
<section class="cards">{cards}</section>
<section class="panel"><h2>Telemetry</h2><figure class="chart">{telemetry_svg}</figure></section>
{loss_panel}
<section class="tables">
<div class="panel"><h2>Results</h2>{_table(summary_rows(df, summary, trace))}</div>
<div><div class="panel"><h2>Workload</h2>{_table(workload_rows)}</div>
<div class="panel"><h2>Environment</h2>{_table(environment)}</div></div>
</section>
<footer>Schema v{_e(report.get('schema_version'))} · exported with {files}</footer>
</main>
</body>
</html>
"""
