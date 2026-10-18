import os
from datetime import datetime

import pandas as pd

from app.utils.plots import build_report_figure


class BenchmarkLogger:
    def __init__(self):
        self.data = []
        self.is_logging = False

    def start(self):
        """Clears old data and starts listening for new telemetry."""
        self.data = []
        self.is_logging = True

    def log(self, telemetry_data: dict):
        """Appends a row of data if a benchmark is currently running."""
        if self.is_logging:
            self.data.append(telemetry_data)

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
    def export(df: pd.DataFrame, output_dir: str = "logs") -> tuple[str, str]:
        """Writes the collected telemetry to CSV and generates a plot PNG.

        Returns (csv_path, plot_path).
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"run_{timestamp_str}.csv")
        plot_path = os.path.join(output_dir, f"plot_{timestamp_str}.png")

        df.to_csv(csv_path, index=False)
        BenchmarkLogger._generate_plot(df, plot_path)

        return csv_path, plot_path

    @staticmethod
    def _generate_plot(df, save_path):
        build_report_figure(df).savefig(save_path, dpi=300)
