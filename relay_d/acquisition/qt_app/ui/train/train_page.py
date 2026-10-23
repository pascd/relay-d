"""Train tab: pick a dataset folder, adjust the main hyperparameters, start training in a terminal.

The work is done by ``relay_d.training`` (also available as ``relay-d-train``). Runs live in their own
terminal and keep going when this window is closed; the table at the bottom follows them from disk.
"""

import os
import sys
import threading
import time

from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QAbstractItemView,
)

from relay_d.training import launcher, runs
from relay_d.utils import user_settings
from relay_d.utils.coloring_logger import logger
from ...utils.custom_message_box import CustomMessageBox
from ..settings.config_manager import config

_FROM_FOLDER = "From the folder's config"  # robomimic: use robomimic_config.json / postprocess options
_REFRESH_MS = 2000  # run table refresh


class TrainPage:
    def __init__(self, ui_instance, settings_page=None):
        self.ui = ui_instance
        self.settings_page = settings_page
        self.datasets = []
        self.folder = ""
        self._param_widgets = {}
        self._busy = 0  # background jobs running (environment check, stopping a run)
        self._build_ui()
        self.refresh_runs()

        self._timer = QTimer(self.ui.page_train)
        self._timer.timeout.connect(self.refresh_runs)
        self._timer.start(_REFRESH_MS)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QVBoxLayout(self.ui.page_train)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        title = QLabel("Train a Policy")
        title.setObjectName("trainPageTitle")
        root.addWidget(title)

        # Dataset
        box = QGroupBox("Dataset")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.edit_folder = QLineEdit()
        self.edit_folder.setReadOnly(True)
        self.edit_folder.setPlaceholderText("Folder written by Data Post Process (or the dataset itself)")
        self.btn_folder = QPushButton("Choose Folder...")
        self.btn_folder.clicked.connect(self.choose_folder)
        row.addWidget(self.edit_folder, 1)
        row.addWidget(self.btn_folder)
        lay.addLayout(row)
        self.combo_dataset = QComboBox()
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        lay.addWidget(self.combo_dataset)
        self.label_summary = QLabel("No dataset selected")
        self.label_summary.setWordWrap(True)
        lay.addWidget(self.label_summary)
        root.addWidget(box)

        # Policy and parameters
        box = QGroupBox("Policy and parameters")
        lay = QVBoxLayout(box)
        self.form = QFormLayout()
        self.combo_policy = QComboBox()
        self.combo_policy.currentIndexChanged.connect(self._on_policy_changed)
        self.form.addRow("Policy", self.combo_policy)
        lay.addLayout(self.form)
        self.label_hint = QLabel("")
        lay.addWidget(self.label_hint)
        self.edit_extra = QLineEdit()
        self.edit_extra.setPlaceholderText("Extra arguments, added to the training command as written")
        lay.addWidget(self.edit_extra)
        root.addWidget(box)

        # Python environment
        box = QGroupBox("Python environment")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.label_python = QLabel("Python")
        self.edit_python = QLineEdit()
        self.edit_python.editingFinished.connect(self._save_python)
        self.btn_browse_python = QPushButton("Browse...")
        self.btn_browse_python.clicked.connect(self.choose_python)
        self.btn_check = QPushButton("Check environment")
        self.btn_check.clicked.connect(lambda: self.check_environment())
        for w in (self.label_python, self.edit_python):
            row.addWidget(w, 1 if w is self.edit_python else 0)
        row.addWidget(self.btn_browse_python)
        row.addWidget(self.btn_check)
        lay.addLayout(row)
        self.label_env = QLabel("")
        self.label_env.setWordWrap(True)
        lay.addWidget(self.label_env)
        root.addWidget(box)

        self.btn_start = QPushButton("Start Training")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_training)
        root.addWidget(self.btn_start)

        # Runs
        box = QGroupBox("Runs")
        lay = QVBoxLayout(box)
        self.table = QTableWidget(0, 4)
        self.table.setMinimumHeight(140)
        self.table.setHorizontalHeaderLabels(["Run", "Status", "Started", "Command"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        for text, slot in (("Stop", self.stop_run), ("Follow log in terminal", self.follow_log),
                           ("Open run folder", self.open_run_folder)):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn)
            if slot == self.stop_run:
                self.btn_stop = btn
        row.addStretch(1)
        lay.addLayout(row)
        root.addWidget(box, 1)

    # ------------------------------------------------------------------ dataset

    def choose_folder(self):
        start = self.folder or config.get("hdf5_save_folder") or os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self.ui, "Choose the dataset folder", start)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder):
        self.folder = folder
        self.edit_folder.setText(folder)
        self.datasets = launcher.find_datasets(folder)
        self.combo_dataset.blockSignals(True)
        self.combo_dataset.clear()
        for d in self.datasets:
            self.combo_dataset.addItem(f"{d.format}: {d.path.name}")
        self.combo_dataset.blockSignals(False)
        self.combo_dataset.setVisible(len(self.datasets) > 1)
        if not self.datasets:
            self.label_summary.setText("No robomimic file (with a mask group) or LeRobot dataset found in this folder.")
            self.btn_start.setEnabled(False)
        self._on_dataset_changed()
        self.refresh_runs()

    def _dataset(self):
        i = self.combo_dataset.currentIndex()
        return self.datasets[i] if 0 <= i < len(self.datasets) else None

    def _on_dataset_changed(self):
        ds = self._dataset()
        self.btn_start.setEnabled(ds is not None)
        self.combo_policy.blockSignals(True)
        self.combo_policy.clear()
        if ds is not None:
            if ds.format == launcher.ROBOMIMIC:
                self.combo_policy.addItems([_FROM_FOLDER, *launcher.ROBOMIMIC_POLICIES])
            else:
                self.combo_policy.addItems(launcher.LEROBOT_POLICIES)
            s = ds.summary
            parts = [f"{s.get('episodes')} episodes", f"{s.get('frames')} frames"]
            if ds.format == launcher.LEROBOT:
                parts += [f"{s.get('fps')} fps", f"{len(s.get('cameras', []))} camera(s)"]
            elif not s.get("has_valid"):
                parts.append("no validation split (validation is turned off)")
            if ds.format == launcher.LEROBOT and not s.get("cameras"):
                parts.append("NO CAMERA: ACT and Diffusion need at least one image input")
            self.label_summary.setText(f"{ds.format}: {ds.path}\n" + ", ".join(parts))
            self.edit_python.setText(config.get(f"python_{ds.format}") or sys.executable)
            self.label_python.setText(f"Python with {ds.format}")
            self.label_env.setText("")
        self.combo_policy.blockSignals(False)
        self._on_policy_changed()

    # ------------------------------------------------------------------ parameters

    def _policy(self):
        text = self.combo_policy.currentText()
        return None if text in ("", _FROM_FOLDER) else text

    def _on_policy_changed(self):
        for widget, label in self._param_widgets.values():
            self.form.removeRow(label)
        self._param_widgets = {}
        ds = self._dataset()
        if ds is None:
            return
        try:
            defaults = launcher.default_params(ds, self._policy())
        except Exception as e:  # a broken saved config must not break the page
            logger.error(f"Could not read the default training parameters: {e}")
            CustomMessageBox.error(str(e), title="Training configuration")
            return
        spec = launcher.ROBOMIMIC_PARAMS if ds.format == launcher.ROBOMIMIC else launcher.LEROBOT_PARAMS
        for key, (_where, kind, text) in spec.items():
            value = defaults.get(key)
            if value is None:
                continue  # the saved config does not define it: leave it alone
            if kind is int:
                widget = QSpinBox()
                widget.setRange(0 if key in launcher._ZERO_OK else 1, 2_000_000_000)
                widget.setValue(int(value))
            elif kind is float:
                widget = QDoubleSpinBox()
                widget.setDecimals(8)
                widget.setSingleStep(1e-5)
                widget.setRange(1e-8, 10.0)
                widget.setValue(float(value))
            else:
                widget = QComboBox()
                widget.addItems(["cuda", "cpu"])
                widget.setCurrentText(str(value))
            label = QLabel(text)
            self.form.addRow(label, widget)
            self._param_widgets[key] = (widget, label)
            if key in ("steps", "batch_size") and ds.format == launcher.LEROBOT:
                widget.valueChanged.connect(self._update_hint)
        self._update_hint()

    def _params(self):
        out = {}
        for key, (widget, _label) in self._param_widgets.items():
            out[key] = widget.currentText() if isinstance(widget, QComboBox) else widget.value()
        return out

    def _update_hint(self):
        ds = self._dataset()
        p = self._params()
        frames = (ds.summary.get("frames") if ds else None) or 0
        if ds and ds.format == launcher.LEROBOT and frames and "steps" in p and "batch_size" in p:
            self.label_hint.setText(f"LeRobot counts steps: about {p['steps'] * p['batch_size'] / frames:.1f} epochs.")
        else:
            self.label_hint.setText("")

    # ------------------------------------------------------------------ python environment

    def _save_python(self):
        ds = self._dataset()
        if ds is not None:
            config.save_settings({f"python_{ds.format}": self.edit_python.text().strip()})

    def choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "Choose a Python interpreter", self.edit_python.text())
        if path:
            self.edit_python.setText(path)
            self._save_python()

    def _run_in_background(self, work, on_done):
        """Run ``work()`` in a thread and call ``on_done(value, error)`` from the GUI thread when it ends.

        Importing torch or waiting for a training process to exit takes seconds; on the GUI thread that would
        also stop the ROS spin timer (a running recording would hold and repeat its last samples).
        """
        outcome = {}

        def target():
            try:
                outcome["value"] = work()
            except Exception as e:  # reported to on_done, not lost with the thread
                outcome["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._set_busy(+1)
        timer = QTimer(self.ui.page_train)
        timer.setInterval(100)

        def poll():
            if thread.is_alive():
                return
            timer.stop()
            timer.deleteLater()
            self._set_busy(-1)
            on_done(outcome.get("value"), outcome.get("error"))

        timer.timeout.connect(poll)
        timer.start()

    def _set_busy(self, delta):
        self._busy += delta
        idle = self._busy == 0
        self.btn_check.setEnabled(idle)
        self.btn_stop.setEnabled(idle)
        self.btn_start.setEnabled(idle and self._dataset() is not None)
        # What was checked is what starts: no changing the dataset, policy, parameters or Python meanwhile.
        inputs = [self.btn_folder, self.combo_dataset, self.combo_policy, self.edit_python,
                  self.btn_browse_python, self.edit_extra]
        for widget in inputs + [w for w, _label in self._param_widgets.values()]:
            widget.setEnabled(idle)

    def check_environment(self, then=None, ds=None, python=None):
        """Look for the training package in a Python interpreter (in the background).

        Defaults to the dataset and Python shown now. ``then(ok)`` is called when the check is done.
        """
        ds = ds or self._dataset()
        if ds is None:
            return
        python = python or self.edit_python.text().strip() or sys.executable
        self.label_env.setText("Checking the environment (importing torch can take a few seconds)...")

        def done(result, error):
            if error is not None:
                result = {"ok": False, "error": str(error), "cuda": False, "torch": None}
            if result["ok"]:
                self.label_env.setText(f"OK: {ds.format} found, torch {result['torch']}, "
                                       f"CUDA {'available' if result['cuda'] else 'not available'}")
            else:
                self.label_env.setText(f"Not usable: {result['error']}")
            if then is not None:
                then(result["ok"])

        self._run_in_background(lambda: launcher.check_environment(python, ds.format), done)

    # ------------------------------------------------------------------ start

    def start_training(self):
        ds = self._dataset()
        if ds is None:
            return
        self._save_python()
        # The job is built now, from what is on screen at the click, and that is what gets checked and started.
        job = launcher.TrainJob(ds, self.edit_python.text().strip() or sys.executable, self._policy(),
                                self._params(), self.edit_extra.text())
        terminal = user_settings.get("train_use_terminal")
        self.check_environment(then=lambda ok: self._start_after_check(ok, job, terminal), ds=ds, python=job.python)

    def _start_after_check(self, ok, job, terminal):
        if not ok:
            CustomMessageBox.error(self.label_env.text(), title="Cannot start training")
            return

        def done(run_dir, error):
            if error is not None:  # TrainingError is a bad parameter or dataset; anything else is unexpected
                CustomMessageBox.error(str(error), title="Cannot start training")
                return
            logger.info(f"Started training run {run_dir}")
            if runs.read_run(run_dir).meta.get("terminal") is None:
                CustomMessageBox.info(
                    f"No terminal window could be opened, so training runs in the background.\nLog: {run_dir / 'train.log'}",
                    title="Training started",
                )
            self.refresh_runs(select=run_dir.name)

        # Opening a terminal can take a moment (the launcher waits to see that it did start): not on the GUI thread.
        self._run_in_background(lambda: launcher.start(job, terminal=terminal), done)

    # ------------------------------------------------------------------ runs

    def refresh_runs(self, select=None):
        found = runs.list_runs(self.folder) if self.folder else []
        selected = select or self._selected_name()
        self.table.setRowCount(len(found))
        for row, r in enumerate(found):
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.meta.get("started", 0)))
            for col, text in enumerate((r.path.name, r.status, started, r.meta.get("command", ""))):
                self.table.setItem(row, col, QTableWidgetItem(text))
            if r.path.name == selected:
                self.table.selectRow(row)
        self._runs = found

    def _selected_name(self):
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else None

    def _selected_run(self):
        row = self.table.currentRow()
        found = getattr(self, "_runs", [])
        if 0 <= row < len(found):
            return found[row]
        CustomMessageBox.info("Select a run in the table first.", title="Runs")
        return None

    def stop_run(self):
        run = self._selected_run()
        if run is None or run.status not in (runs.RUNNING, runs.STARTING):
            return
        if not user_settings.get("confirm_destructive") or CustomMessageBox.question(
            f"Stop {run.path.name}?", title="Stop training"
        ):
            def done(stopped, _error):
                if not stopped:
                    CustomMessageBox.error("The run did not stop; close its terminal window.", title="Stop training")
                self.refresh_runs()

            self._run_in_background(lambda: runs.stop(run.path), done)

    def follow_log(self):
        run = self._selected_run()
        if run is not None and not launcher.reopen_terminal(run.path):
            CustomMessageBox.info(f"No terminal available. The log is at:\n{run.log}", title="Log")

    def open_run_folder(self):
        run = self._selected_run()
        if run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(run.path)))
