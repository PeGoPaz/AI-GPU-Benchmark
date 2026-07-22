import time
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt6.QtCore import QThread, pyqtSignal


class HeavySyntheticModel(nn.Module):
    """A heavy neural network layer stack designed to stress GPU compute and VRAM."""
    def __init__(self, size=4096):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 10)
        )

    def forward(self, x):
        return self.layers(x)


class TrainerWorker(QThread):
    # Signals to communicate training progress back to the UI
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, float)  # (current_step, loss)
    training_finished = pyqtSignal(dict)       # Summary metrics
    error_occurred = pyqtSignal(str)

    def __init__(self, steps: int = 1000, batch_size: int = 256, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.batch_size = batch_size
        self._is_running = True

    def run(self):
        """Executes the GPU training workload in a separate thread."""
        if not torch.cuda.is_available():
            self.error_occurred.emit("PyTorch cannot detect CUDA/GPU!")
            return

        try:
            device = torch.device("cuda:0")
            self.status_updated.emit(f"Allocating model & tensors on {torch.cuda.get_device_name(0)}...")

            # 1. Initialize heavy synthetic benchmark model & optimizer
            model = HeavySyntheticModel(size=8192).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            start_time = time.time()
            total_loss = 0.0

            self.status_updated.emit("Benchmark training started — stress testing GPU...")

            # 2. Main Training Loop
            for step in range(1, self.steps + 1):
                if not self._is_running:
                    self.status_updated.emit("Training aborted by user.")
                    return

                # Generate dummy high-dimensional batch on CUDA
                inputs = torch.randn(self.batch_size, 8192, device=device)
                targets = torch.randn(self.batch_size, 10, device=device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                total_loss += loss_val

                # Emit step progress
                if step % 10 == 0 or step == self.steps:
                    self.progress_updated.emit(step, loss_val)

            elapsed_time = time.time() - start_time
            avg_loss = total_loss / self.steps

            summary = {
                "total_steps": self.steps,
                "elapsed_time_sec": round(elapsed_time, 2),
                "avg_loss": round(avg_loss, 4),
                "steps_per_sec": round(self.steps / elapsed_time, 2),
            }

            self.status_updated.emit("Benchmark run complete!")
            self.training_finished.emit(summary)

        except Exception as e:
            self.error_occurred.emit(f"Training Error: {str(e)}")

    def stop(self):
        """Safely interrupts the training loop."""
        self._is_running = False
        self.wait()