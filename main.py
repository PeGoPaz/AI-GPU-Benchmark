import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import pynvml

def check_gpu():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return f"{gpu_name} Detected | Temp: {temp}°C"
    except Exception as e:
        return f"GPU Error: {str(e)}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("AI GPU Benchmark - Environment Test")
    
    layout = QVBoxLayout()
    label = QLabel(check_gpu())
    layout.addWidget(label)
    
    window.setLayout(layout)
    window.resize(400, 150)
    window.show()
    
    sys.exit(app.exec())