import os

from PyQt6.QtCore import QByteArray, QSettings, QSize, Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QAction, QActionGroup, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gpu_bench.core.telemetry import TelemetryWorker
from gpu_bench.core.trainer import (
    DEFAULT_PRESET,
    MODELS,
    PRECISIONS,
    PRESETS,
    SEQ_LENGTHS,
    VRAM_RESERVE_GIB,
    TrainerWorker,
    estimate_weights_gib,
    fits_in_vram,
)
from gpu_bench.ui import icons, readings
from gpu_bench.ui.about import AboutDialog, about_rows, app_icon
from gpu_bench.ui.compare_window import CompareWindow, load_runs
from gpu_bench.ui.live_charts import LiveCharts
from gpu_bench.ui.style import (
    LIGHT,
    START_GREEN,
    STOP_RED,
    THEMES,
    build_qpalette,
    build_stylesheet,
)
from gpu_bench.ui.widgets import header
from gpu_bench.ui.widgets.empty_state import EmptyState
from gpu_bench.ui.widgets.gauge import ArcGauge
from gpu_bench.ui.widgets.header import HeaderBar
from gpu_bench.ui.widgets.phase_progress import PhaseProgressBar
from gpu_bench.ui.widgets.result_card import ResultCard
from gpu_bench.ui.widgets.tile import MetricTile
from gpu_bench.utils.display import RateTracker, format_progress, headline_cards, summary_rows
from gpu_bench.utils.logger import BenchmarkLogger
from gpu_bench.utils.report import build_report

SETTINGS_ORG = SETTINGS_APP = "AI-GPU-Benchmark"
SYSTEM = "system"

# How often the live charts take in new data: as often as telemetry arrives.
# pyqtgraph swaps curve data without redrawing the figure, so this is cheap.
LIVE_REFRESH_MS = 250


def _mirror(button: QPushButton, action: QAction) -> None:
    """Makes a button a second face of an action: same trigger, same state."""
    button.clicked.connect(action.trigger)
    button.setToolTip(action.toolTip())
    button.setEnabled(action.isEnabled())
    action.changed.connect(lambda: button.setEnabled(action.isEnabled()))


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        # Injectable so tests write to a scratch file instead of the user's
        # real preferences.
        self.settings = settings if settings is not None else QSettings(
            SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle("AI GPU Benchmark — detecting GPU…")
        self.setWindowIcon(app_icon())
        self.resize(1360, 900)              # until a saved geometry says otherwise

        # Main Layout: the header across the top, the two panels below it.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.header = HeaderBar()
        outer_layout.addWidget(self.header)
        # Left | right, resizable; the left column scrolls when it outgrows
        # the window instead of forcing the window to grow.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        self.main_splitter.setChildrenCollapsible(False)
        outer_layout.addWidget(self.main_splitter, 1)

        # --- LEFT PANEL: Hardware Stats & Controls ---
        left_panel = QVBoxLayout()

        # 1. Hardware Dashboard: gauges for the readings with a full scale,
        # tiles for the rest, in readings.GAUGES / readings.TILES order.
        hw_group = QGroupBox("Live Hardware Telemetry")
        hw_layout = QVBoxLayout()
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(4)
        # Power and VRAM have no scale until the card reports its limit and
        # total; until then they grow to fit what has been seen.
        scales = {"temp": readings.TEMP_SCALE_C, "util": readings.UTIL_SCALE_PCT}
        self.gauges = {}
        for key, title, unit in readings.GAUGES:
            self.gauges[key] = ArcGauge(title, unit, scales.get(key))
            gauge_row.addWidget(self.gauges[key])
        hw_layout.addLayout(gauge_row)

        tile_grid = QGridLayout()
        tile_grid.setSpacing(8)
        self.tiles = {}
        for i, (key, title, _, trend) in enumerate(readings.TILES):
            tile = MetricTile(title, trend)
            self.tiles[key] = tile
            tile_grid.addWidget(tile, i // 2, i % 2)
        self.tiles["throttle"].setToolTip(
            "Why the clock is below its maximum, straight from the driver: "
            "power cap, power brake, or thermal slowdown."
        )
        self.tiles["throttle"].lbl_value.setWordWrap(True)
        hw_layout.addLayout(tile_grid)
        hw_group.setLayout(hw_layout)
        left_panel.addWidget(hw_group)

        # 2. Workload settings
        self.workload_group = QGroupBox("Workload")
        workload_layout = QFormLayout()

        # Fills in every field below; editing any of them by hand turns the
        # selection into Custom, and matching a preset again turns it back.
        self.cmb_preset = QComboBox()
        for i, preset in enumerate(PRESETS):
            self.cmb_preset.addItem(preset.label, preset)
            self.cmb_preset.setItemData(i, preset.description,
                                        Qt.ItemDataRole.ToolTipRole)
        self.cmb_preset.addItem("Custom", None)

        self.cmb_model = QComboBox()
        for i, spec in enumerate(MODELS):
            self.cmb_model.addItem(spec.label, spec)
            self.cmb_model.setItemData(
                i, f"{spec.model_id}\n{spec.params_b:.2f}B parameters",
                Qt.ItemDataRole.ToolTipRole)

        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(10, 5000)
        self.spin_steps.setSingleStep(10)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)

        self.cmb_precision = QComboBox()
        for spec in PRECISIONS:
            self.cmb_precision.addItem(spec.label, spec)
        self.cmb_precision.setToolTip(
            "BF16 needs Ampere or newer. FP32 runs anywhere but roughly "
            "doubles VRAM and time. INT8/INT4 store the frozen base model "
            "quantized (bitsandbytes) — what fits 3B–7B models on an 8 GB card."
        )

        self.cmb_seq_len = QComboBox()
        for length in SEQ_LENGTHS:
            self.cmb_seq_len.addItem(f"{length} tokens", length)
        self.cmb_seq_len.setToolTip(
            "Tokens per training sequence. Compute and activation memory grow "
            "with it, attention quadratically — the main lever on run length."
        )

        self.chk_checkpointing = QCheckBox("Gradient checkpointing")
        self.chk_checkpointing.setToolTip(
            "Recomputes activations during the backward pass instead of storing "
            "them: far less VRAM, roughly a third more compute per step. Needed "
            "for long sequences on small cards."
        )

        self.spin_warmup = QSpinBox()
        self.spin_warmup.setRange(0, 100)
        self.spin_warmup.setValue(10)
        self.spin_warmup.setToolTip(
            "Steps run before the timer starts, to get model loading and CUDA "
            "kernel compilation out of the measured window."
        )

        workload_layout.addRow("Preset:", self.cmb_preset)
        workload_layout.addRow("Model:", self.cmb_model)
        workload_layout.addRow("Steps:", self.spin_steps)
        workload_layout.addRow("Batch size:", self.spin_batch)
        workload_layout.addRow("Precision:", self.cmb_precision)
        workload_layout.addRow("Sequence length:", self.cmb_seq_len)
        workload_layout.addRow("Warm-up steps:", self.spin_warmup)
        workload_layout.addRow(self.chk_checkpointing)

        # Updated whenever model or precision changes, and once the card's
        # VRAM size is known from the first telemetry snapshot.
        self.lbl_fit = QLabel()
        self.lbl_fit.setObjectName("fitHint")
        self.lbl_fit.setWordWrap(True)
        workload_layout.addRow(self.lbl_fit)
        self.workload_group.setLayout(workload_layout)
        left_panel.addWidget(self.workload_group)

        # 3. Controls
        ctrl_group = QGroupBox("Benchmark Controls")
        ctrl_layout = QVBoxLayout()

        # One QAction per command: the toolbar, the shortcut and the button
        # below all trigger it, and its enabled state is the only one there
        # is — the buttons mirror it rather than being switched separately.
        self.act_start = QAction("Start", self)
        self.act_start.setShortcut(QKeySequence("Ctrl+R"))
        self.act_start.setToolTip("Start the benchmark (Ctrl+R)")
        self.act_start.triggered.connect(self.start_benchmark)
        self.act_stop = QAction("Stop", self)
        self.act_stop.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.act_stop.setToolTip("Stop at the next step boundary (Esc)")
        self.act_stop.triggered.connect(self.stop_benchmark)
        self.act_stop.setEnabled(False)
        self.act_export = QAction("Export", self)
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.setToolTip("Write CSV, PNG and the run report to logs/ (Ctrl+E)")
        self.act_export.triggered.connect(self.export_results)
        self.act_export.setEnabled(False)
        self.act_open_report = QAction("Open report", self)
        self.act_open_report.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_report.setToolTip("Open the last exported HTML report in the browser (Ctrl+O)")
        self.act_open_report.triggered.connect(self.open_report)
        self.act_open_report.setEnabled(False)
        self._last_html = None

        self.btn_start = QPushButton("Start Stress Test")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setMinimumHeight(40)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setMinimumHeight(35)
        self.btn_export = QPushButton("Export Results")
        self.btn_export.setObjectName("btnExport")
        self.btn_export.setMinimumHeight(35)
        for button, action in ((self.btn_start, self.act_start),
                               (self.btn_stop, self.act_stop),
                               (self.btn_export, self.act_export)):
            _mirror(button, action)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_export)
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)
        left_panel.addStretch()

        # --- RIGHT PANEL: Console, Progress, Plots, Summary ---
        right_panel = QVBoxLayout()

        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.log_to_console("System initialized. Ready for benchmark.")

        self.phase_bar = PhaseProgressBar()

        self.lbl_progress = QLabel("")
        self.lbl_progress.setObjectName("progressInfo")

        # Live charts: telemetry (two plots on one time axis) and loss — or,
        # before the first run, a note on how to get some.
        self.charts = LiveCharts()
        self.charts.setMinimumHeight(320)
        self.empty_charts = EmptyState(
            "No run yet",
            "Pick a preset and press Start (Ctrl+R) — live charts appear here. "
            "The telemetry on the left is already live.",
            app_icon().pixmap(56, 56))
        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(self.empty_charts)
        self.chart_stack.addWidget(self.charts)

        # Summary stats table
        self.summary_group = QGroupBox("Benchmark Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table.setMaximumHeight(260)

        # The headline numbers first; every row of the table on request.
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self.result_cards = [ResultCard() for _ in range(5)]
        for card in self.result_cards:
            cards_row.addWidget(card)
        self.btn_details = QToolButton()
        self.btn_details.setObjectName("detailsToggle")
        self.btn_details.setText("Details")
        self.btn_details.setCheckable(True)
        self.btn_details.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_details.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_details.toggled.connect(self._toggle_details)
        self.summary_table.setVisible(False)

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addLayout(cards_row)
        results_layout.addWidget(self.btn_details)
        results_layout.addWidget(self.summary_table)
        self.empty_summary = EmptyState("No results yet",
                                        "Results appear here when a run finishes.")
        self.summary_stack = QStackedWidget()
        self.summary_stack.addWidget(self.empty_summary)
        self.summary_stack.addWidget(results_page)
        self._results_page = results_page
        summary_layout.addWidget(self.summary_stack)
        self.summary_group.setLayout(summary_layout)

        # Progress on top; below it charts, summary and log share the height
        # through a splitter, the log last — it is the least read of the three
        # and can be dragged shut or hidden from the View menu.
        self.log_panel = QWidget()
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Execution Log:"))
        log_layout.addWidget(self.console)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setObjectName("rightSplitter")
        self.right_splitter.addWidget(self.chart_stack)
        self.right_splitter.addWidget(self.summary_group)
        self.right_splitter.addWidget(self.log_panel)
        self.right_splitter.setCollapsible(0, False)
        # Spare height goes to the charts; the summary and the log take what
        # their content asks for. The summary appearing at the end of a run
        # would otherwise squeeze the charts to make room for itself.
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 0)
        self.right_splitter.setStretchFactor(2, 0)
        self.summary_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                         QSizePolicy.Policy.Maximum)
        self.console.setMinimumHeight(60)
        self._default_split_pending = True     # cleared when a saved split is restored

        right_panel.addWidget(self.phase_bar)
        right_panel.addWidget(self.lbl_progress)
        right_panel.addWidget(self.right_splitter, 1)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("leftScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(360)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([440, 920])

        # Menu: the theme is a preference, not a benchmark setting.
        file_menu = self.menuBar().addMenu("&File")
        self.act_compare = QAction("&Compare runs…", self)
        self.act_compare.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_compare.setToolTip("Open several exported run reports side by side")
        self.act_compare.triggered.connect(self.compare_runs)
        file_menu.addAction(self.act_compare)
        self._compare_windows = []

        run_menu = self.menuBar().addMenu("&Run")
        for action in (self.act_start, self.act_stop, self.act_export, self.act_open_report):
            run_menu.addAction(action)
        view_menu = self.menuBar().addMenu("&View")
        # Theme: follow the system by default, or pin light or dark.
        self.theme_menu = view_menu.addMenu("&Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self.theme_actions = {}
        for choice, label in ((SYSTEM, "&System"), ("light", "&Light"), ("dark", "&Dark")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, c=choice: self.set_theme_choice(c))
            theme_group.addAction(action)
            self.theme_menu.addAction(action)
            self.theme_actions[choice] = action
        self.act_toggle_theme = QAction("Switch light/dark", self)
        self.act_toggle_theme.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self.act_toggle_theme.triggered.connect(self._toggle_theme)
        self.theme_menu.addSeparator()
        self.theme_menu.addAction(self.act_toggle_theme)
        self._theme_choice = SYSTEM
        hints = QGuiApplication.styleHints()
        if hints is not None and hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        self.act_log = QAction("Execution &log", self)
        self.act_log.setCheckable(True)
        self.act_log.setChecked(True)
        self.act_log.setShortcut(QKeySequence("Ctrl+L"))
        self.act_log.toggled.connect(self.log_panel.setVisible)
        view_menu.addAction(self.act_log)

        help_menu = self.menuBar().addMenu("&Help")
        self.act_about = QAction("&About AI GPU Benchmark", self)
        self.act_about.setMenuRole(QAction.MenuRole.AboutRole)
        self.act_about.triggered.connect(self.show_about)
        help_menu.addAction(self.act_about)

        self.toolbar = QToolBar("Benchmark")
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.setIconSize(QSize(18, 18))
        for action in (self.act_start, self.act_stop, self.act_export, self.act_open_report):
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_log)
        self.theme_button = QToolButton()
        self.theme_button.setText("Theme")
        self.theme_button.setToolTip("Theme: follow the system, or pin light or dark")
        self.theme_button.setMenu(self.theme_menu)
        self.theme_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.theme_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.addWidget(self.theme_button)
        self.addToolBar(self.toolbar)

        # Applied last, so every widget above is polished in one pass.
        self._theme = LIGHT
        self._apply_theme(LIGHT)

        self._vram_total_mib = None          # MiB, from the first snapshot
        self.cmb_model.currentIndexChanged.connect(self._update_fit_hint)
        self.cmb_precision.currentIndexChanged.connect(self._update_fit_hint)

        self._applying_preset = False
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_selected)
        for signal in (self.cmb_model.currentIndexChanged,
                       self.spin_steps.valueChanged,
                       self.spin_batch.valueChanged,
                       self.cmb_precision.currentIndexChanged,
                       self.cmb_seq_len.currentIndexChanged,
                       self.chk_checkpointing.toggled):
            signal.connect(self._sync_preset_combo)
        self._restore_settings()
        self._update_fit_hint()

        # --- INITIALIZE THREADS ---
        self.logger = BenchmarkLogger()
        self._last_telemetry_error = None
        self.telemetry = TelemetryWorker(interval_ms=250)
        self.telemetry.data_updated.connect(self.update_telemetry_ui)
        self.telemetry.gpu_name_ready.connect(self._on_gpu_name_ready)
        self.telemetry.device_info_ready.connect(self._on_device_info)
        self.telemetry.sampler_status.connect(self._on_sampler_status)
        self.telemetry.power_trace.connect(self._on_power_trace)
        self.telemetry.error_occurred.connect(self._on_telemetry_error)
        self.telemetry.start()

        self.trainer = None
        self._device_info = None
        self._sampler_status = None
        self._last_summary = None

        # Live charts: redrawn from what the logger has collected so far.
        self._loss_steps, self._loss_values = [], []
        self._run_warmup = 0
        self._total_steps = 0
        self._rate = RateTracker()
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(LIVE_REFRESH_MS)
        self._live_timer.timeout.connect(self._refresh_live_charts)

    def log_to_console(self, text: str):
        """Appends text to the console window."""
        self.console.append(text)

    @pyqtSlot(str)
    def _on_gpu_name_ready(self, gpu_name: str):
        """Updates the window title once the GPU name is detected."""
        self.setWindowTitle(f"AI GPU Benchmark — {gpu_name}")
        self.header.set_gpu_name(gpu_name)
        self.log_to_console(f"Detected GPU: {gpu_name}")

    @pyqtSlot(dict)
    def _on_device_info(self, info: dict):
        self._device_info = info
        self.header.set_device_info(info)
        if info.get("driver_version"):
            self.log_to_console(
                f"Driver {info['driver_version']}, CUDA {info.get('cuda_driver_version') or '?'}")

    @pyqtSlot(dict)
    def _on_sampler_status(self, status: dict):
        self._sampler_status = status
        self.header.set_sampler(status)
        if status["active"]:
            self.log_to_console(f"Power sampler: native, {status['detail']}, "
                                f"every {status['interval_us'] / 1000:g} ms")
        else:
            self.log_to_console(f"Power sampler: {status['detail']}")

    @pyqtSlot(object)
    def _on_power_trace(self, chunk):
        self.logger.log_power_trace(chunk)
        if self.logger.is_logging:
            self.charts.append_trace(chunk)

    @pyqtSlot(str)
    def _on_telemetry_error(self, message: str):
        """Surfaces NVML failures instead of leaving the dashboard on '--'.

        Read errors repeat on every poll, so an identical consecutive message
        is swallowed rather than filling the console with the same line four
        times a second.
        """
        if message == self._last_telemetry_error:
            return
        self._last_telemetry_error = message
        if message.startswith("NVML Initialization Failed"):
            self.header.set_gpu_unavailable(message)
            self.empty_charts.set_text(
                "No NVIDIA GPU detected",
                "Runs need an NVIDIA card and driver. The presets, the VRAM estimate "
                "and File → Compare runs (Ctrl+Shift+O) work without one.")
        self.log_to_console(f"TELEMETRY ERROR: {message}")

    @pyqtSlot(dict)
    def update_telemetry_ui(self, data: dict):
        """Updates the dashboard with live PyNVML data."""
        self.logger.log(data)
        if self.logger.is_logging:
            self.charts.append(data)
        if self._vram_total_mib is None and data.get('vram_total_mb'):
            self._vram_total_mib = data['vram_total_mb']
            self._update_fit_hint()
        self._show_gauges(data)
        for key, _, read, _ in readings.TILES:
            self.tiles[key].set_reading(read(data))

    def _show_gauges(self, data: dict):
        limit = (self._device_info or {}).get("power_limit_w")
        if limit:
            self.gauges["power"].set_maximum(limit)
        if data.get("vram_total_mb"):
            self.gauges["vram"].set_maximum(data["vram_total_mb"] / 1024)

        for key, reading in readings.gauge_readings(data, limit).items():
            number = reading.number
            if key == "vram" and number is not None:
                number /= 1024                    # the gauge is scaled in GiB
            self.gauges[key].set_value(number, reading.value, reading.state, reading.caption)

    # --- theme & persisted settings --------------------------------------------

    def _apply_theme(self, theme):
        self._theme = theme
        app = QApplication.instance()
        if isinstance(app, QApplication):
            # App-wide rather than on the window: combo popups and tooltips
            # are top-level windows and would not inherit it.
            app.setPalette(build_qpalette(theme, self.style().standardPalette()))
        self.setStyleSheet(build_stylesheet(theme))
        self._paint_icons(theme)
        self.phase_bar.set_theme(theme)
        self.header.set_theme(theme)
        self.charts.set_theme(theme)
        for gauge in self.gauges.values():
            gauge.set_theme(theme)
        for tile in self.tiles.values():
            tile.set_theme(theme)

    def about_dialog(self) -> AboutDialog:
        return AboutDialog(about_rows(self._device_info, self._sampler_status), self)

    def show_about(self):
        self.about_dialog().exec()

    def _paint_icons(self, theme):
        """Toolbar glyphs in the theme's text colour; Start and Stop in theirs."""
        text = theme.text_strong
        for action, glyph, color in ((self.act_start, "start", START_GREEN),
                                     (self.act_stop, "stop", STOP_RED),
                                     (self.act_export, "export", text),
                                     (self.act_open_report, "report", text),
                                     (self.act_log, "log", text),
                                     (self.act_toggle_theme, "theme", text)):
            action.setIcon(icons.icon(glyph, color))
        self.theme_button.setIcon(icons.icon("theme", text))

    @staticmethod
    def system_is_dark() -> bool:
        """The OS colour scheme, where Qt can tell (6.5+); light otherwise."""
        hints = QGuiApplication.styleHints()
        if hints is None or not hasattr(hints, "colorScheme"):
            return False
        return hints.colorScheme() == Qt.ColorScheme.Dark

    def theme_choice(self) -> str:
        return self._theme_choice

    def set_theme_choice(self, choice: str):
        """System, light or dark; System re-resolves when the OS changes."""
        if choice not in self.theme_actions:
            choice = SYSTEM
        self._theme_choice = choice
        self.theme_actions[choice].setChecked(True)
        name = ("dark" if self.system_is_dark() else "light") if choice == SYSTEM else choice
        self._apply_theme(THEMES[name])
        self.settings.setValue("ui/theme", choice)

    def _on_system_scheme_changed(self, *_):
        if self._theme_choice == SYSTEM:
            self.set_theme_choice(SYSTEM)

    def _toggle_theme(self):
        """Ctrl+Shift+D: the other theme, pinned — it overrides System."""
        self.set_theme_choice("light" if self._theme.name == "dark" else "dark")

    def _save_settings(self):
        """Remembers the workload, so the next launch opens on the last run."""
        s = self.settings
        s.setValue("ui/theme", self._theme_choice)
        s.setValue("workload/preset", self.cmb_preset.currentText())
        s.setValue("workload/model", self.cmb_model.currentText())
        s.setValue("workload/precision", self.cmb_precision.currentText())
        s.setValue("workload/steps", self.spin_steps.value())
        s.setValue("workload/batch_size", self.spin_batch.value())
        s.setValue("workload/seq_len", self.cmb_seq_len.currentData())
        s.setValue("workload/gradient_checkpointing", self.chk_checkpointing.isChecked())
        s.setValue("workload/warmup_steps", self.spin_warmup.value())
        s.setValue("ui/geometry", self.saveGeometry())
        s.setValue("ui/main_splitter", self.main_splitter.saveState())
        s.setValue("ui/right_splitter", self.right_splitter.saveState())
        s.setValue("ui/show_log", self.act_log.isChecked())
        s.sync()

    def _setting_int(self, key: str, default: int) -> int:
        """An int from the settings file, or the default if it is not one."""
        try:
            return self.settings.value(key, default, type=int)
        except (TypeError, ValueError):
            return default

    def _restore_settings(self):
        """Reapplies saved settings, skipping anything that no longer exists.

        Values are matched by label, so a model dropped from the registry or a
        hand-edited file falls back to the default instead of raising.
        """
        s = self.settings
        preset_label = s.value("workload/preset", DEFAULT_PRESET.label, type=str)
        preset = next((p for p in PRESETS if p.label == preset_label), None)
        self._apply_preset(preset or DEFAULT_PRESET)

        if preset_label == "Custom":
            self._applying_preset = True
            try:
                for combo, key in ((self.cmb_model, "workload/model"),
                                   (self.cmb_precision, "workload/precision")):
                    index = combo.findText(s.value(key, "", type=str))
                    if index >= 0:
                        combo.setCurrentIndex(index)
                seq_index = self.cmb_seq_len.findData(
                    self._setting_int("workload/seq_len", 0))
                if seq_index >= 0:
                    self.cmb_seq_len.setCurrentIndex(seq_index)
                self.spin_steps.setValue(
                    self._setting_int("workload/steps", self.spin_steps.value()))
                self.spin_batch.setValue(
                    self._setting_int("workload/batch_size", self.spin_batch.value()))
                self.chk_checkpointing.setChecked(
                    s.value("workload/gradient_checkpointing",
                            self.chk_checkpointing.isChecked(), type=bool))
            finally:
                self._applying_preset = False
            self._sync_preset_combo()

        self.spin_warmup.setValue(
            self._setting_int("workload/warmup_steps", self.spin_warmup.value()))
        # "light" and "dark" from before System existed mean what they say.
        self.set_theme_choice(s.value("ui/theme", SYSTEM, type=str))
        self._restore_layout()

    def _restore_layout(self):
        """Window size and panel splits as they were left. A missing or
        corrupt value is ignored: the defaults set in __init__ stand."""
        s = self.settings
        for key, restore in (("ui/geometry", self.restoreGeometry),
                             ("ui/main_splitter", self.main_splitter.restoreState),
                             ("ui/right_splitter", self.right_splitter.restoreState)):
            value = s.value(key)
            if isinstance(value, QByteArray) and restore(value) and key == "ui/right_splitter":
                self._default_split_pending = False
        self.act_log.setChecked(s.value("ui/show_log", True, type=bool))

    def _apply_preset(self, preset):
        """Writes a preset into the fields without bouncing back to Custom."""
        self._applying_preset = True
        try:
            self.cmb_model.setCurrentIndex(MODELS.index(preset.model))
            self.cmb_precision.setCurrentIndex(PRECISIONS.index(preset.precision))
            self.spin_steps.setValue(preset.steps)
            self.spin_batch.setValue(preset.batch_size)
            self.cmb_seq_len.setCurrentIndex(SEQ_LENGTHS.index(preset.seq_len))
            self.chk_checkpointing.setChecked(preset.gradient_checkpointing)
        finally:
            self._applying_preset = False
        self._sync_preset_combo()

    def _matching_preset(self):
        """The preset the fields currently spell out, or None for Custom."""
        current = (self.cmb_model.currentData(), self.cmb_precision.currentData(),
                   self.spin_steps.value(), self.spin_batch.value(),
                   self.cmb_seq_len.currentData(), self.chk_checkpointing.isChecked())
        for preset in PRESETS:
            if current == (preset.model, preset.precision, preset.steps,
                           preset.batch_size, preset.seq_len,
                           preset.gradient_checkpointing):
                return preset
        return None

    def _on_preset_selected(self, index: int):
        preset = self.cmb_preset.itemData(index)
        # Picking Custom by hand keeps whatever the fields already hold.
        if preset is not None:
            self._apply_preset(preset)

    def _sync_preset_combo(self):
        if self._applying_preset:
            return
        preset = self._matching_preset()
        index = len(PRESETS) if preset is None else PRESETS.index(preset)
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.setCurrentIndex(index)
        self.cmb_preset.blockSignals(False)

    def _update_fit_hint(self):
        """Shows the weights' footprint against the card, flagging a misfit.

        A warning rather than a lock: the estimate covers weights only, and
        someone may well want to watch a run hit the wall on purpose.
        """
        model = self.cmb_model.currentData()
        precision = self.cmb_precision.currentData()
        weights = estimate_weights_gib(model, precision)

        if self._vram_total_mib is None:
            text, misfit = f"Weights ≈ {weights:.1f} GiB", False
        else:
            total = self._vram_total_mib / 1024
            misfit = not fits_in_vram(model, precision, self._vram_total_mib)
            text = f"Weights ≈ {weights:.1f} GiB of {total:.1f} GiB VRAM"
            if misfit:
                text += (f" — too big: a run needs ~{VRAM_RESERVE_GIB:.0f} GiB "
                         "free on top of the weights. Try INT4 or a smaller model.")

        self.lbl_fit.setText(text)
        # A dynamic property drives the stylesheet; Qt only re-reads it on a
        # re-polish, so the colour would otherwise stick to the first state.
        self.lbl_fit.setProperty("warning", misfit)
        self.lbl_fit.style().unpolish(self.lbl_fit)
        self.lbl_fit.style().polish(self.lbl_fit)

    def start_benchmark(self):
        """Disables UI and fires up the PyTorch training thread."""
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.act_export.setEnabled(False)
        self.phase_bar.start(self.spin_warmup.value(), self.spin_steps.value())
        self.header.set_status(header.LOADING)
        self.empty_summary.set_text("Running…", "Results appear here when the run finishes.")
        self.summary_stack.setCurrentWidget(self.empty_summary)
        self._loss_steps, self._loss_values = [], []
        self._rate = RateTracker()
        self._run_warmup = self.spin_warmup.value()
        self.lbl_progress.setText("")
        # Visible from the start and fed as data arrives: a run on the
        # heavier presets is far too long to watch a bare progress bar.
        self.charts.reset()
        self.charts.set_warmup(self._run_warmup)
        self.chart_stack.setCurrentWidget(self.charts)
        self._live_timer.start()
        self.log_to_console("\n--- Starting Synthetic Workload ---")
        if self.lbl_fit.property("warning"):
            self.log_to_console(f"WARNING: {self.lbl_fit.text()}")
        self.logger.start()

        # Locked for the duration: the summary reports these values, so they
        # must describe the run that actually happened.
        self.workload_group.setEnabled(False)
        self._save_settings()

        self.trainer = TrainerWorker(
            steps=self.spin_steps.value(),
            batch_size=self.spin_batch.value(),
            model=self.cmb_model.currentData(),
            warmup_steps=self.spin_warmup.value(),
            precision=self.cmb_precision.currentData(),
            seq_len=self.cmb_seq_len.currentData(),
            gradient_checkpointing=self.chk_checkpointing.isChecked(),
        )
        self.trainer.status_updated.connect(self.log_to_console)
        self.trainer.progress_updated.connect(self._on_progress)
        self.trainer.loss_updated.connect(self._on_loss)
        self.trainer.max_steps_ready.connect(self._on_max_steps)
        self.trainer.training_finished.connect(self.on_benchmark_finished)
        self.trainer.error_occurred.connect(self.on_benchmark_error)
        self.trainer.start()

    def stop_benchmark(self):
        """Requests an early stop; the run ends at the next step boundary."""
        if self.trainer and self.trainer.isRunning():
            self.log_to_console("\nStop requested — finishing current step...")
            self.trainer.stop()
            self.header.set_status(header.STOPPING)
        self.act_stop.setEnabled(False)

    @pyqtSlot(int)
    def _on_max_steps(self, total: int):
        self._total_steps = total
        self.phase_bar.set_total(total)

    @pyqtSlot(int)
    def _on_progress(self, step: int):
        """Moves the bar and recomputes rate and ETA."""
        self.phase_bar.set_step(step)
        if self.header.status() != header.STOPPING:
            self.header.set_status(header.WARMUP if self.phase_bar.phase() == "warm-up"
                                   else header.MEASURING)
        rate = self._rate.update(step)
        self.lbl_progress.setText(format_progress(step, self._total_steps, rate))

    @pyqtSlot(int, float)
    def _on_loss(self, step: int, loss: float):
        self._loss_steps.append(step)
        self._loss_values.append(loss)
        self.charts.append_loss(step, loss)

    def _refresh_live_charts(self):
        self.charts.refresh()

    @pyqtSlot(dict)
    def on_benchmark_finished(self, summary: dict):
        self._live_timer.stop()
        self.phase_bar.finish()
        self.header.set_status(header.STOPPED if summary.get('aborted') else header.DONE)
        self.act_stop.setEnabled(False)
        self._last_summary = summary
        df = self.logger.stop()
        if df is not None:
            self._render_plots(df, summary)
        if summary.get('aborted'):
            self.log_to_console(
                f"\nBenchmark stopped early: {summary['total_steps']}"
                f"/{summary['requested_steps']} steps in {summary['elapsed_time_sec']}s"
            )
        else:
            self.log_to_console(f"\nBenchmark Complete! Time: {summary['elapsed_time_sec']}s")
        self.log_to_console(f"Throughput: {summary['steps_per_sec']} steps/sec")
        if summary.get('tokens_per_step') and summary['steps_per_sec']:
            self.log_to_console(
                f"           {summary['steps_per_sec'] * summary['tokens_per_step']:,.0f} tokens/sec")
        self._set_idle_controls()
        self.act_export.setEnabled(True)

    @pyqtSlot(str)
    def on_benchmark_error(self, message: str):
        """Recovers the UI from a failed run.

        TrainerWorker.run() emits error_occurred and returns without ever
        emitting training_finished, so nothing else re-enables the controls:
        before this, any training failure left Start greyed out for good and
        the app had to be restarted.
        """
        self._live_timer.stop()
        self._last_summary = None
        self.log_to_console(f"ERROR: {message}")
        # Close the run, or its partial telemetry leaks into the next one.
        df = self.logger.stop()
        self.phase_bar.reset()
        self.header.set_status(header.FAILED)
        self.empty_summary.set_text("The run failed", "No results — the log says why.")
        self._set_idle_controls()
        self.act_export.setEnabled(df is not None)

    def _set_idle_controls(self):
        """Returns the control panel to its ready-for-a-new-run state."""
        self.workload_group.setEnabled(True)
        self.act_stop.setEnabled(False)
        self.act_start.setEnabled(True)

    def _render_plots(self, df, summary: dict):
        """Final draw from the complete run, exactly as it will be exported."""
        self.charts.show_run(df, self.logger.power_trace())
        self.chart_stack.setCurrentWidget(self.charts)
        self.charts.setCurrentIndex(0)

        # --- Summary Stats Table ---
        self._render_summary(df, summary)
        self.log_to_console("Summary and plots rendered.")

    def _toggle_details(self, shown: bool):
        self.summary_table.setVisible(shown)
        self.btn_details.setArrowType(Qt.ArrowType.DownArrow if shown
                                      else Qt.ArrowType.RightArrow)

    def _render_summary(self, df, summary: dict):
        """Headline cards, and the full table under Details."""
        trace = self.logger.power_trace()
        headlines = headline_cards(df, summary, trace)
        for card, headline in zip(self.result_cards, headlines + [None] * 5, strict=False):
            card.setVisible(headline is not None)
            if headline is not None:
                card.set_headline(headline)

        metrics = summary_rows(df, summary, trace)

        self.summary_table.setRowCount(len(metrics))
        for row, (metric, value) in enumerate(metrics):
            item_m = QTableWidgetItem(metric)
            item_m.setFlags(item_m.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item_v = QTableWidgetItem(value)
            item_v.setFlags(item_v.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.summary_table.setItem(row, 0, item_m)
            self.summary_table.setItem(row, 1, item_v)

        self.summary_stack.setCurrentWidget(self._results_page)

    def export_results(self):
        """Exports the last benchmark's telemetry to CSV and renders plot PNG."""
        df = self.logger.to_dataframe()
        if df is None:
            self.log_to_console("No data to export.")
            return
        trace = self.logger.power_trace()
        report = None
        if self._last_summary is not None:
            report = build_report(df, self._last_summary, self._device_info,
                                  (self._loss_steps, self._loss_values),
                                  trace, self._sampler_status)
        try:
            paths = BenchmarkLogger.export(df, report=report, trace=trace,
                                           summary=self._last_summary)
        except OSError as e:
            self.log_to_console(f"Export failed: {e}")
            return
        for path in paths:
            if path:
                self.log_to_console(f"Exported: {path}")
        if paths.html:
            self._last_html = paths.html
            self.act_open_report.setEnabled(True)

    def compare_runs(self, paths=None):
        """Opens a comparison of the chosen reports; asks for them if not given."""
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Compare runs", "logs", "Run reports (report_*.json);;JSON (*.json)")
        if not paths:
            return None
        runs, errors = load_runs(list(paths))
        if not runs:
            self.log_to_console("Compare runs: " + "; ".join(errors))
            return None
        window = CompareWindow(runs, errors, self._theme)
        # Kept, or Python would collect the window the moment this returns.
        self._compare_windows.append(window)
        window.show()
        return window

    def open_report(self):
        if self._last_html:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self._last_html)))

    def showEvent(self, event):
        """First show without a saved layout: most of the height to the charts.

        Done here rather than in __init__ because a splitter only knows its
        real height once the window is on screen.
        """
        super().showEvent(event)
        if self._default_split_pending:
            self._default_split_pending = False
            total = max(1, sum(self.right_splitter.sizes()))
            summary = self.summary_group.sizeHint().height()
            log = int(total * 0.18)
            self.right_splitter.setSizes([total - summary - log, summary, log])

    def closeEvent(self, event):
        """Clean up threads on exit."""
        self._live_timer.stop()
        self._save_settings()
        self.telemetry.stop()
        if self.trainer and self.trainer.isRunning():
            # Shutdown is the one place we genuinely must block, so the
            # training thread cannot outlive the window it reports into.
            self.trainer.wait_for_exit()
        event.accept()
