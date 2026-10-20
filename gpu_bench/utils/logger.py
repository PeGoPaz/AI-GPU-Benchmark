import json
import os
from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd

from gpu_bench.utils.plots import build_report_figure


class ExportPaths(NamedTuple):
    csv: str
    plot: str
    report: str | None = None
    power_trace: str | None = None   # only with the native sampler
    html: str | None = None          # the human-readable page, with a report


class BenchmarkLogger:
    def __init__(self):
        self.data = []
        self.trace_chunks = []
        self.is_logging = False

    def start(self):
        """Clears old data and starts listening for new telemetry."""
        self.data = []
        self.trace_chunks = []
        self.is_logging = True

    def log(self, telemetry_data: dict):
        """Appends a row of data if a benchmark is currently running."""
        if self.is_logging:
            self.data.append(telemetry_data)

    def log_power_trace(self, chunk) -> None:
        """Keeps a (seconds, watts) batch from the native sampler while logging.

        Stored as the arrays it arrived in and only joined on request: at
        500 Hz a long run is hundreds of thousands of samples, too many to
        turn into dicts one by one as they come in.
        """
        if self.is_logging:
            self.trace_chunks.append(chunk)

    def power_trace(self) -> pd.DataFrame | None:
        """The high-rate power trace, on the same time axis as to_dataframe().

        None when the native sampler was not running.
        """
        if not self.trace_chunks:
            return None
        seconds = np.concatenate([chunk[0] for chunk in self.trace_chunks])
        watts = np.concatenate([chunk[1] for chunk in self.trace_chunks])
        start = self.data[0]['timestamp'] if self.data else seconds[0]
        return pd.DataFrame({
            'timestamp': seconds,
            'power_w': watts,
            'time_sec': seconds - start,
        })

    def stop(self) -> pd.DataFrame | None:
        """Stops logging and returns the collected run as a DataFrame."""
        self.is_logging = False
        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame | None:
        """Returns the collected telemetry without changing logging state.

        Safe to call repeatedly — exporting must not depend on whether stop()
        happened to be called first, nor mutate the run being inspected.
        """
        if not self.data:
            return None

        df = pd.DataFrame(self.data)

        # Normalize time so the X-axis starts at 0 seconds
        start_time = df['timestamp'].iloc[0]
        df['time_sec'] = df['timestamp'] - start_time

        return df

    @staticmethod
    def export(df: pd.DataFrame, output_dir: str = "logs",
               report: dict | None = None,
               trace: pd.DataFrame | None = None,
               summary: dict | None = None) -> ExportPaths:
        """Writes the collected telemetry to CSV and generates a plot PNG.

        With a report (see gpu_bench.utils.report), also writes it as JSON beside
        them, pointing at the other files. A run that failed has no summary
        to report, so its telemetry is still exportable without one. With the
        native sampler's trace, that goes into a CSV of its own — at 500 Hz
        it has thousands of rows for every one of the telemetry CSV's. With
        the run's summary as well, the report is also written as a
        self-contained HTML page (gpu_bench.utils.html_report).
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"run_{timestamp_str}.csv")
        plot_path = os.path.join(output_dir, f"plot_{timestamp_str}.png")

        df.to_csv(csv_path, index=False)
        BenchmarkLogger._generate_plot(df, plot_path, trace)

        trace_path = None
        if trace is not None and len(trace):
            trace_path = os.path.join(output_dir, f"power_{timestamp_str}.csv")
            trace.to_csv(trace_path, index=False)

        report_path = html_path = None
        if report is not None:
            report_path = os.path.join(output_dir, f"report_{timestamp_str}.json")
            files = {"csv": os.path.basename(csv_path), "plot": os.path.basename(plot_path)}
            if trace_path:
                files["power_trace"] = os.path.basename(trace_path)
            if summary is not None:
                html_path = os.path.join(output_dir, f"report_{timestamp_str}.html")
                files["html"] = os.path.basename(html_path)
            report = {**report, "files": files}
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            if html_path is not None and summary is not None:
                # Imported here: matplotlib's SVG backend and the display
                # helpers are only needed when there is a page to write.
                from gpu_bench.utils.html_report import build_html
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(build_html(report, df, summary, trace))

        return ExportPaths(csv_path, plot_path, report_path, trace_path, html_path)

    @staticmethod
    def _generate_plot(df, save_path, trace=None):
        build_report_figure(df, trace).savefig(save_path, dpi=300)
