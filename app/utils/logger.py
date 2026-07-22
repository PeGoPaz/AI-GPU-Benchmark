import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class BenchmarkLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
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

    def stop_and_save(self) -> str:
        """Stops logging, saves to CSV, and generates a graph. Returns plot path."""
        self.is_logging = False
        
        if not self.data:
            return None

        # Convert to Pandas DataFrame
        df = pd.DataFrame(self.data)
        
        # Normalize time so the X-axis starts at 0 seconds
        start_time = df['timestamp'].iloc[0]
        df['time_sec'] = df['timestamp'] - start_time

        # Generate unique filenames based on current time
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.log_dir, f"run_{timestamp_str}.csv")
        plot_path = os.path.join(self.log_dir, f"plot_{timestamp_str}.png")

        # Save raw data for your GitHub repo
        df.to_csv(csv_path, index=False)

        # Generate portfolio-ready visual
        self._generate_plot(df, plot_path)

        return plot_path

    def _generate_plot(self, df, save_path):
        plt.figure(figsize=(10, 5))
        
        # Left Y-Axis: GPU Temperature (Red)
        ax1 = plt.gca()
        ax1.plot(df['time_sec'], df['temp_gpu'], color='#ff4c4c', label='GPU Temp (°C)', linewidth=2)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Temperature (°C)', color='#ff4c4c')
        ax1.tick_params(axis='y', labelcolor='#ff4c4c')

        # Right Y-Axis: Power Draw (Blue)
        ax2 = ax1.twinx()
        ax2.plot(df['time_sec'], df['power_w'], color='#4c72ff', label='Power Draw (W)', linewidth=2, linestyle='--')
        ax2.set_ylabel('Power (W)', color='#4c72ff')
        ax2.tick_params(axis='y', labelcolor='#4c72ff')

        plt.title('GPU Thermal & Power Curve During Stress Test')
        plt.grid(True, alpha=0.2)
        
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300) # High-res for X.com/Portfolio
        plt.close()