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
        """Stops logging and returns the collected data as a DataFrame."""
        self.is_logging = False

        if not self.data:
            return None

        df = pd.DataFrame(self.data)

        # Normalize time so the X-axis starts at 0 seconds
        start_time = df['timestamp'].iloc[0]
        df['time_sec'] = df['timestamp'] - start_time

        return df