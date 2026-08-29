import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


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
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 7))

        # --- Top: Thermal & Power ---
        ax1.plot(df['time_sec'], df['temp_gpu'], color='#ff4c4c',
                 label='GPU Temp (°C)', linewidth=2)
        ax1.set_ylabel('Temperature (°C)', color='#ff4c4c')
        ax1.tick_params(axis='y', labelcolor='#ff4c4c')

        ax2 = ax1.twinx()
        ax2.plot(df['time_sec'], df['power_w'], color='#4c72ff',
                 label='Power Draw (W)', linewidth=2, linestyle='--')
        ax2.set_ylabel('Power (W)', color='#4c72ff')
        ax2.tick_params(axis='y', labelcolor='#4c72ff')
        ax1.set_title('Thermal & Power Curve')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.grid(True, alpha=0.2)

        # --- Bottom: Utilization & Clock ---
        ax3.plot(df['time_sec'], df['gpu_util_pct'], color='#ffa500',
                 label='GPU Utilization (%)', linewidth=2)
        ax3.plot(df['time_sec'], df['sm_clock_mhz'], color='#00bfff',
                 label='SM Clock (MHz)', linewidth=2, linestyle='--')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Utilization (%)')
        ax3.set_title('Utilization & Clock Speed')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
