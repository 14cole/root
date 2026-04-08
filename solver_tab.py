"""
Solver UI threshold guide.

Quality gate inputs:
- `residual <=`: max allowed linear solve residual norm (dimensionless).
- `cond <=`: max allowed condition-number estimate of the system matrix (dimensionless).
- `warns <=`: max allowed warning count emitted by the solver.
- `Require quality gate pass`: fail the solve if any quality threshold is exceeded.

Mesh convergence inputs:
- `fine factor`: panel-density multiplier for the fine mesh solve (`> 1.0`).
- `rms dB <=`: max allowed RMS of `(base_rcs_db - fine_rcs_db)` over all sample points.
- `max |dB| <=`: max allowed worst-case absolute dB delta between base and fine solves.
- `Run mesh convergence check`: execute the additional fine-mesh solve and compute deltas.
- `Require mesh convergence pass`: fail the solve if mesh thresholds are exceeded.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_io import build_geometry_snapshot, parse_geometry
from grim_io import export_result_to_grim
from rcs_solver import solve_monostatic_rcs_2d
from solver_quality import evaluate_mesh_convergence, scale_snapshot_panel_density


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class _SolveWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(
        self,
        snapshot: Dict[str, Any],
        source_path: str,
        base_dir: str,
        frequencies: List[float],
        elevations: List[float],
        pol: str,
        units: str,
        quality_thresholds: Dict[str, float | int],
        strict_quality_gate: bool,
        mesh_convergence: bool,
        mesh_fine_factor: float,
        mesh_rms_limit_db: float,
        mesh_max_abs_limit_db: float,
        strict_mesh_convergence: bool,
    ):
        super().__init__()
        self.snapshot = snapshot
        self.source_path = source_path
        self.base_dir = base_dir
        self.frequencies = frequencies
        self.elevations = elevations
        self.pol = pol
        self.units = units
        self.quality_thresholds = dict(quality_thresholds)
        self.strict_quality_gate = strict_quality_gate
        self.mesh_convergence = bool(mesh_convergence)
        self.mesh_fine_factor = float(mesh_fine_factor)
        self.mesh_rms_limit_db = float(mesh_rms_limit_db)
        self.mesh_max_abs_limit_db = float(mesh_max_abs_limit_db)
        self.strict_mesh_convergence = bool(strict_mesh_convergence)

    def _on_progress(self, done_steps: int, total_steps: int, message: str) -> None:
        if total_steps <= 0:
            pct = 0
        else:
            pct = int(round(100.0 * float(done_steps) / float(total_steps)))
        pct = max(0, min(100, pct))
        self.progress.emit(pct, message)

    @Slot()
    def run(self):
        try:
            if self.mesh_convergence:
                base_progress_scale = 0.6

                def base_cb(done_steps: int, total_steps: int, message: str) -> None:
                    if total_steps <= 0:
                        pct = 0
                    else:
                        pct = int(round(base_progress_scale * 100.0 * float(done_steps) / float(total_steps)))
                    pct = max(0, min(60, pct))
                    self.progress.emit(pct, message)

                result = solve_monostatic_rcs_2d(
                    geometry_snapshot=self.snapshot,
                    frequencies_ghz=self.frequencies,
                    elevations_deg=self.elevations,
                    polarization=self.pol,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    progress_callback=base_cb,
                    quality_thresholds=self.quality_thresholds,
                    strict_quality_gate=self.strict_quality_gate,
                    compute_condition_number=self.strict_quality_gate,
                )
                self.progress.emit(61, "Running mesh convergence fine-mesh solve...")
                fine_snapshot = scale_snapshot_panel_density(self.snapshot, self.mesh_fine_factor)

                def fine_cb(done_steps: int, total_steps: int, message: str) -> None:
                    if total_steps <= 0:
                        pct = 60
                    else:
                        frac = 0.4 * float(done_steps) / float(total_steps)
                        pct = 60 + int(round(100.0 * frac))
                    pct = max(60, min(100, pct))
                    self.progress.emit(pct, f"Mesh check: {message}")

                fine_result = solve_monostatic_rcs_2d(
                    geometry_snapshot=fine_snapshot,
                    frequencies_ghz=self.frequencies,
                    elevations_deg=self.elevations,
                    polarization=self.pol,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    progress_callback=fine_cb,
                    quality_thresholds=self.quality_thresholds,
                    strict_quality_gate=False,
                    compute_condition_number=False,
                )
                mesh_gate = evaluate_mesh_convergence(
                    base_result=result,
                    fine_result=fine_result,
                    rms_limit_db=self.mesh_rms_limit_db,
                    max_abs_limit_db=self.mesh_max_abs_limit_db,
                )
                mesh_gate["fine_factor"] = self.mesh_fine_factor
                metadata = dict(result.get("metadata", {}) or {})
                metadata["mesh_convergence"] = mesh_gate
                result["metadata"] = metadata

                if self.strict_mesh_convergence and not bool(mesh_gate.get("passed", False)):
                    reason = str(mesh_gate.get("reason", "") or "mesh convergence failed")
                    raise ValueError(f"Mesh convergence gate failed: {reason}")
            else:
                result = solve_monostatic_rcs_2d(
                    geometry_snapshot=self.snapshot,
                    frequencies_ghz=self.frequencies,
                    elevations_deg=self.elevations,
                    polarization=self.pol,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    progress_callback=self._on_progress,
                    quality_thresholds=self.quality_thresholds,
                    strict_quality_gate=self.strict_quality_gate,
                    compute_condition_number=self.strict_quality_gate,
                )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result, self.source_path)


class SolverTab(QWidget):
    def __init__(self, geometry_tab=None, parent=None):
        super().__init__(parent)
        self.geometry_tab = geometry_tab
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_source_path: str = ""
        self._solve_thread: Optional[QThread] = None
        self._solve_worker: Optional[_SolveWorker] = None
        self._is_solving: bool = False

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 700])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        self._update_mode_enables()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        geometry_group = QGroupBox("Geometry Source")
        geometry_grid = QGridLayout(geometry_group)
        self.edit_geo_path = QLineEdit()
        self.edit_geo_path.setPlaceholderText("Optional: use .geo file directly, or use current Geometry tab")
        self.btn_browse_geo = QPushButton("Browse...")
        self.btn_use_tab = QPushButton("Use Geometry Tab")
        self.lbl_geo = QLabel("No explicit file selected. Solver will use Geometry tab if available.")
        self.lbl_geo.setWordWrap(True)
        geometry_grid.addWidget(QLabel("Geometry File"), 0, 0)
        geometry_grid.addWidget(self.edit_geo_path, 0, 1, 1, 2)
        geometry_grid.addWidget(self.btn_browse_geo, 1, 1)
        geometry_grid.addWidget(self.btn_use_tab, 1, 2)
        geometry_grid.addWidget(self.lbl_geo, 2, 0, 1, 3)
        layout.addWidget(geometry_group)

        options_group = QGroupBox("Solve Options")
        options_form = QFormLayout(options_group)

        self.cmb_units = QComboBox()
        self.cmb_units.addItems(["inches", "meters"])
        self.cmb_units.setCurrentText("inches")

        self.cmb_pol = QComboBox()
        self.cmb_pol.addItem("TE (vertical / VV)", userData="TE")
        self.cmb_pol.addItem("TM (horizontal / HH)", userData="TM")
        self.chk_strict_quality = QCheckBox("Require quality gate pass")
        self.chk_strict_quality.setChecked(False)
        self.edit_quality_residual_max = QLineEdit("1e-2")
        self.edit_quality_condition_max = QLineEdit("1e6")
        self.edit_quality_warnings_max = QLineEdit("10")
        quality_threshold_row = QWidget()
        quality_threshold_layout = QHBoxLayout(quality_threshold_row)
        quality_threshold_layout.setContentsMargins(0, 0, 0, 0)
        quality_threshold_layout.addWidget(QLabel("residual<="))
        quality_threshold_layout.addWidget(self.edit_quality_residual_max)
        quality_threshold_layout.addWidget(QLabel("cond<="))
        quality_threshold_layout.addWidget(self.edit_quality_condition_max)
        quality_threshold_layout.addWidget(QLabel("warns<="))
        quality_threshold_layout.addWidget(self.edit_quality_warnings_max)

        self.chk_mesh_convergence = QCheckBox("Run mesh convergence check")
        self.chk_mesh_convergence.setChecked(False)
        self.chk_strict_mesh = QCheckBox("Require mesh convergence pass")
        self.chk_strict_mesh.setChecked(False)
        self.edit_mesh_fine_factor = QLineEdit("1.5")
        self.edit_mesh_rms_max_db = QLineEdit("1.0")
        self.edit_mesh_max_abs_max_db = QLineEdit("3.0")
        mesh_threshold_row = QWidget()
        mesh_threshold_layout = QHBoxLayout(mesh_threshold_row)
        mesh_threshold_layout.setContentsMargins(0, 0, 0, 0)
        mesh_threshold_layout.addWidget(QLabel("fine factor"))
        mesh_threshold_layout.addWidget(self.edit_mesh_fine_factor)
        mesh_threshold_layout.addWidget(QLabel("rms dB<="))
        mesh_threshold_layout.addWidget(self.edit_mesh_rms_max_db)
        mesh_threshold_layout.addWidget(QLabel("max |dB|<="))
        mesh_threshold_layout.addWidget(self.edit_mesh_max_abs_max_db)

        self.cmb_freq_mode = QComboBox()
        self.cmb_freq_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_freq_list = QLineEdit("1.0, 3.0, 10.0")
        self.edit_freq_start = QLineEdit("1.0")
        self.edit_freq_stop = QLineEdit("10.0")
        self.edit_freq_step = QLineEdit("1.0")

        self.cmb_elev_mode = QComboBox()
        self.cmb_elev_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_elev_list = QLineEdit("0, 30, 60, 90, 120, 150, 180")
        self.edit_elev_start = QLineEdit("0")
        self.edit_elev_stop = QLineEdit("180")
        self.edit_elev_step = QLineEdit("2")

        freq_sweep_row = QWidget()
        freq_sweep_layout = QHBoxLayout(freq_sweep_row)
        freq_sweep_layout.setContentsMargins(0, 0, 0, 0)
        freq_sweep_layout.addWidget(QLabel("Start"))
        freq_sweep_layout.addWidget(self.edit_freq_start)
        freq_sweep_layout.addWidget(QLabel("Stop"))
        freq_sweep_layout.addWidget(self.edit_freq_stop)
        freq_sweep_layout.addWidget(QLabel("Step"))
        freq_sweep_layout.addWidget(self.edit_freq_step)

        elev_sweep_row = QWidget()
        elev_sweep_layout = QHBoxLayout(elev_sweep_row)
        elev_sweep_layout.setContentsMargins(0, 0, 0, 0)
        elev_sweep_layout.addWidget(QLabel("Start"))
        elev_sweep_layout.addWidget(self.edit_elev_start)
        elev_sweep_layout.addWidget(QLabel("Stop"))
        elev_sweep_layout.addWidget(self.edit_elev_stop)
        elev_sweep_layout.addWidget(QLabel("Step"))
        elev_sweep_layout.addWidget(self.edit_elev_step)

        options_form.addRow("Units In Geometry", self.cmb_units)
        options_form.addRow("Polarization", self.cmb_pol)
        options_form.addRow("Quality Gate", self.chk_strict_quality)
        options_form.addRow("Quality Thresholds", quality_threshold_row)
        options_form.addRow("Mesh Check", self.chk_mesh_convergence)
        options_form.addRow("Mesh Strict", self.chk_strict_mesh)
        options_form.addRow("Mesh Thresholds", mesh_threshold_row)
        options_form.addRow("Frequency Mode", self.cmb_freq_mode)
        options_form.addRow("Frequencies (GHz)", self.edit_freq_list)
        options_form.addRow("Frequency Sweep", freq_sweep_row)
        options_form.addRow("Elevation Mode", self.cmb_elev_mode)
        options_form.addRow("Elevations (deg)", self.edit_elev_list)
        options_form.addRow("Elevation Sweep", elev_sweep_row)
        layout.addWidget(options_group)

        output_group = QGroupBox("Output")
        output_grid = QGridLayout(output_group)
        self.edit_output = QLineEdit("rcs_output.grim")
        self.btn_browse_output = QPushButton("Browse...")
        self.chk_export_after_solve = QCheckBox("Export .grim automatically after solve")
        self.chk_export_after_solve.setChecked(True)
        output_grid.addWidget(QLabel("GRIM Output"), 0, 0)
        output_grid.addWidget(self.edit_output, 0, 1)
        output_grid.addWidget(self.btn_browse_output, 0, 2)
        output_grid.addWidget(self.chk_export_after_solve, 1, 0, 1, 3)
        layout.addWidget(output_group)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Monostatic Solver")
        self.btn_export = QPushButton("Export Last Result")
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

        self.btn_browse_geo.clicked.connect(self._browse_geo)
        self.btn_use_tab.clicked.connect(self._use_geometry_tab)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_run.clicked.connect(self._run_solver)
        self.btn_export.clicked.connect(self._export_last_result)
        self.cmb_freq_mode.currentIndexChanged.connect(self._update_mode_enables)
        self.cmb_elev_mode.currentIndexChanged.connect(self._update_mode_enables)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.canvas = MplCanvas(panel)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=2)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(
            ["Frequency (GHz)", "Elevation (deg)", "RCS (linear)", "RCS (dB)"]
        )
        layout.addWidget(self.table_results, stretch=1)
        return panel

    def _browse_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        self.edit_geo_path.setText(fname)
        self.lbl_geo.setText(f"Using geometry file: {fname}")

    def _use_geometry_tab(self):
        self.edit_geo_path.clear()
        if self.geometry_tab is None:
            self.lbl_geo.setText("Geometry tab is not connected in this session.")
            return
        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        title = snapshot.get("title", "Geometry")
        if segment_count <= 0:
            self.lbl_geo.setText("Geometry tab has no loaded segments yet.")
            return
        self.lbl_geo.setText(f"Using Geometry tab: {title} ({segment_count} segment(s)).")

    def _browse_output(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Select Output .grim", "rcs_output.grim", "GRIM Files (*.grim);;All Files (*)"
        )
        if not fname:
            return
        self.edit_output.setText(fname)

    def _update_mode_enables(self):
        freq_discrete = self.cmb_freq_mode.currentIndex() == 0
        elev_discrete = self.cmb_elev_mode.currentIndex() == 0

        self.edit_freq_list.setEnabled(freq_discrete)
        self.edit_freq_start.setEnabled(not freq_discrete)
        self.edit_freq_stop.setEnabled(not freq_discrete)
        self.edit_freq_step.setEnabled(not freq_discrete)

        self.edit_elev_list.setEnabled(elev_discrete)
        self.edit_elev_start.setEnabled(not elev_discrete)
        self.edit_elev_stop.setEnabled(not elev_discrete)
        self.edit_elev_step.setEnabled(not elev_discrete)

    def _parse_list(self, text: str, field_name: str) -> List[float]:
        tokens = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]
        if not tokens:
            raise ValueError(f"{field_name}: no values were provided.")
        values: List[float] = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except ValueError:
                raise ValueError(f"{field_name}: invalid numeric token '{tok}'.")
        return values

    def _parse_sweep(self, start_s: str, stop_s: str, step_s: str, field_name: str) -> List[float]:
        try:
            start = float(start_s)
            stop = float(stop_s)
            step = abs(float(step_s))
        except ValueError:
            raise ValueError(f"{field_name}: start, stop, and step must be numeric.")
        if step <= 0.0:
            raise ValueError(f"{field_name}: step must be > 0.")

        direction = 1.0 if stop >= start else -1.0
        signed_step = step * direction
        values: List[float] = []
        current = start
        for _ in range(20_000):
            if direction > 0 and current > stop + 1e-9:
                break
            if direction < 0 and current < stop - 1e-9:
                break
            values.append(round(current, 12))
            current += signed_step
        if not values or abs(values[-1] - stop) > 1e-9:
            values.append(stop)
        if len(values) > 5000:
            raise ValueError(f"{field_name}: too many samples ({len(values)}).")
        return values

    def _collect_frequency_values(self) -> List[float]:
        if self.cmb_freq_mode.currentIndex() == 0:
            freqs = self._parse_list(self.edit_freq_list.text(), "Frequencies")
        else:
            freqs = self._parse_sweep(
                self.edit_freq_start.text(),
                self.edit_freq_stop.text(),
                self.edit_freq_step.text(),
                "Frequencies",
            )
        if any(f <= 0 for f in freqs):
            raise ValueError("Frequencies must be positive values in GHz.")
        return freqs

    def _collect_elevation_values(self) -> List[float]:
        if self.cmb_elev_mode.currentIndex() == 0:
            return self._parse_list(self.edit_elev_list.text(), "Elevations")
        return self._parse_sweep(
            self.edit_elev_start.text(),
            self.edit_elev_stop.text(),
            self.edit_elev_step.text(),
            "Elevations",
        )

    def _load_geometry_for_solver(self) -> Tuple[Dict[str, Any], str, str]:
        path = self.edit_geo_path.text().strip()
        if path:
            with open(path, "r") as f:
                text = f.read()
            title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
            snapshot = build_geometry_snapshot(title, segments, ibcs_entries, dielectric_entries)
            source_path = os.path.abspath(path)
            base_dir = os.path.dirname(source_path)
            return snapshot, source_path, base_dir

        if self.geometry_tab is None:
            raise ValueError("No geometry file selected and Geometry tab is unavailable.")

        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        if segment_count <= 0:
            raise ValueError("No geometry loaded. Load a .geo file or use the Geometry tab first.")

        source_path = str(snapshot.get("source_path", "") or "")
        if not source_path:
            source_path = str(getattr(self.geometry_tab, "loaded_path", "") or "")
        base_dir = os.path.dirname(source_path) if source_path else os.getcwd()
        return snapshot, source_path, base_dir

    def _set_solving_state(self, solving: bool):
        self._is_solving = solving
        self.btn_run.setEnabled(not solving)
        self.btn_export.setEnabled(not solving)
        self.btn_browse_geo.setEnabled(not solving)
        self.btn_use_tab.setEnabled(not solving)
        self.btn_browse_output.setEnabled(not solving)
        self.edit_geo_path.setEnabled(not solving)
        self.edit_output.setEnabled(not solving)
        self.cmb_units.setEnabled(not solving)
        self.cmb_pol.setEnabled(not solving)
        self.cmb_freq_mode.setEnabled(not solving)
        self.cmb_elev_mode.setEnabled(not solving)
        self.edit_freq_list.setEnabled(not solving and self.cmb_freq_mode.currentIndex() == 0)
        self.edit_elev_list.setEnabled(not solving and self.cmb_elev_mode.currentIndex() == 0)
        self.edit_freq_start.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_freq_stop.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_freq_step.setEnabled(not solving and self.cmb_freq_mode.currentIndex() != 0)
        self.edit_elev_start.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.edit_elev_stop.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.edit_elev_step.setEnabled(not solving and self.cmb_elev_mode.currentIndex() != 0)
        self.chk_export_after_solve.setEnabled(not solving)
        self.chk_strict_quality.setEnabled(not solving)
        self.edit_quality_residual_max.setEnabled(not solving)
        self.edit_quality_condition_max.setEnabled(not solving)
        self.edit_quality_warnings_max.setEnabled(not solving)
        self.chk_mesh_convergence.setEnabled(not solving)
        self.chk_strict_mesh.setEnabled(not solving)
        self.edit_mesh_fine_factor.setEnabled(not solving)
        self.edit_mesh_rms_max_db.setEnabled(not solving)
        self.edit_mesh_max_abs_max_db.setEnabled(not solving)
        self.btn_run.setText("Solving..." if solving else "Run Monostatic Solver")

    @Slot(int, str)
    def _on_solver_progress(self, pct: int, message: str):
        self.progress.setValue(max(0, min(100, int(pct))))
        if message:
            self.lbl_status.setText(f"Solving... {message}")

    @Slot(object, str)
    def _on_solver_finished(self, result: Dict[str, Any], source_path: str):
        self.last_result = result
        self.last_source_path = source_path
        self._populate_results_table(result)
        self._plot_results(result)

        metadata = result.get("metadata", {}) or {}
        freq_count = int(metadata.get("frequency_count", 0))
        elev_count = int(metadata.get("elevation_count", 0))
        units = str(metadata.get("geometry_units_in", self.cmb_units.currentText()))
        pol = str(result.get("solver_polarization", self.cmb_pol.currentData() or "TE"))
        write_summary = (
            f"Solved monostatic 2D RCS with {freq_count} frequency(ies) and "
            f"{elev_count} elevation angle(s)."
        )
        qg = metadata.get("quality_gate", {}) or {}
        quality_suffix = ""
        if isinstance(qg, dict):
            if bool(qg.get("passed", True)):
                quality_suffix = " Quality gate: PASS."
            else:
                violations = qg.get("violations", []) or []
                joined = "; ".join(str(v) for v in violations[:2])
                if len(violations) > 2:
                    joined += f" (+{len(violations) - 2} more)"
                quality_suffix = f" Quality gate: FAIL ({joined})."
        mesh_suffix = ""
        mesh = metadata.get("mesh_convergence", {}) or {}
        if isinstance(mesh, dict) and bool(mesh.get("enabled", False)):
            if bool(mesh.get("passed", False)):
                mesh_suffix = (
                    " Mesh convergence: PASS "
                    f"(rms={float(mesh.get('rms_db', 0.0)):.3g} dB, "
                    f"max={float(mesh.get('max_abs_db', 0.0)):.3g} dB)."
                )
            else:
                reason = str(mesh.get("reason", "") or "criteria not met")
                mesh_suffix = f" Mesh convergence: FAIL ({reason})."
        self.progress.setValue(100)
        self.lbl_status.setText(write_summary + quality_suffix + mesh_suffix)

        if self.chk_export_after_solve.isChecked():
            try:
                out_text = self.edit_output.text().strip() or "rcs_output.grim"
                files = export_result_to_grim(
                    result,
                    out_text,
                    polarization=result.get("polarization", "VV"),
                    source_path=source_path,
                    history=(
                        f"solver=monostatic2d "
                        f"freq_count={freq_count} "
                        f"elev_count={elev_count} "
                        f"units={units} "
                        f"pol={pol}"
                    ),
                )
                self.lbl_status.setText(
                    write_summary
                    + quality_suffix
                    + mesh_suffix
                    + " Exported "
                    + ", ".join(os.path.basename(path) for path in files)
                )
            except Exception as exc:
                QMessageBox.warning(self, "Export Warning", f"Solve completed, but export failed:\n{exc}")

        self._set_solving_state(False)

    @Slot(str)
    def _on_solver_error(self, message: str):
        self.progress.setValue(0)
        self.lbl_status.setText(f"Solve failed: {message}")
        QMessageBox.critical(self, "Solver Error", message)
        self._set_solving_state(False)

    @Slot()
    def _on_solver_thread_finished(self):
        self._solve_worker = None
        self._solve_thread = None

    def _run_solver(self):
        if self._is_solving:
            return
        try:
            frequencies = self._collect_frequency_values()
            elevations = self._collect_elevation_values()
            snapshot, source_path, base_dir = self._load_geometry_for_solver()
            pol = str(self.cmb_pol.currentData())
            units = self.cmb_units.currentText()
            strict_quality = bool(self.chk_strict_quality.isChecked())
            quality_residual_max = float(self.edit_quality_residual_max.text().strip())
            quality_condition_max = float(self.edit_quality_condition_max.text().strip())
            quality_warnings_max = int(float(self.edit_quality_warnings_max.text().strip()))
            if quality_residual_max <= 0.0:
                raise ValueError("Quality residual threshold must be > 0.")
            if quality_condition_max <= 0.0:
                raise ValueError("Quality condition threshold must be > 0.")
            if quality_warnings_max < 0:
                raise ValueError("Quality warning threshold must be >= 0.")
            quality_thresholds: Dict[str, float | int] = {
                "residual_norm_max": quality_residual_max,
                "condition_est_max": quality_condition_max,
                "warnings_max": quality_warnings_max,
            }

            mesh_convergence = bool(self.chk_mesh_convergence.isChecked())
            strict_mesh_convergence = bool(self.chk_strict_mesh.isChecked())
            mesh_fine_factor = float(self.edit_mesh_fine_factor.text().strip())
            mesh_rms_limit_db = float(self.edit_mesh_rms_max_db.text().strip())
            mesh_max_abs_limit_db = float(self.edit_mesh_max_abs_max_db.text().strip())
            if mesh_convergence and mesh_fine_factor <= 1.0:
                raise ValueError("Mesh convergence requires fine factor > 1.0.")
            if mesh_rms_limit_db <= 0.0:
                raise ValueError("Mesh RMS threshold must be > 0.")
            if mesh_max_abs_limit_db <= 0.0:
                raise ValueError("Mesh max-abs threshold must be > 0.")
        except Exception as exc:
            QMessageBox.critical(self, "Solver Error", str(exc))
            self.lbl_status.setText(f"Solve failed: {exc}")
            return

        self.progress.setValue(0)
        self.lbl_status.setText("Starting solver thread...")
        self._set_solving_state(True)

        thread = QThread(self)
        worker = _SolveWorker(
            snapshot=snapshot,
            source_path=source_path,
            base_dir=base_dir,
            frequencies=frequencies,
            elevations=elevations,
            pol=pol,
            units=units,
            quality_thresholds=quality_thresholds,
            strict_quality_gate=strict_quality,
            mesh_convergence=mesh_convergence,
            mesh_fine_factor=mesh_fine_factor,
            mesh_rms_limit_db=mesh_rms_limit_db,
            mesh_max_abs_limit_db=mesh_max_abs_limit_db,
            strict_mesh_convergence=strict_mesh_convergence,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_solver_progress)
        worker.finished.connect(self._on_solver_finished)
        worker.error.connect(self._on_solver_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_solver_thread_finished)

        self._solve_thread = thread
        self._solve_worker = worker
        thread.start()

    def _export_last_result(self):
        if self._is_solving:
            QMessageBox.information(self, "Export", "Solver is currently running. Please wait for completion.")
            return
        if not self.last_result:
            QMessageBox.information(self, "Export", "No solver result exists yet. Run the solver first.")
            return
        out_text = self.edit_output.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Export", "Please provide an output path.")
            return
        try:
            files = export_result_to_grim(
                self.last_result,
                out_text,
                polarization=self.last_result.get("polarization", "VV"),
                source_path=self.last_source_path,
                history="solver=monostatic2d manual_export=1",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        QMessageBox.information(self, "Exported", "\n".join(files))
        self.lbl_status.setText("Exported: " + ", ".join(os.path.basename(path) for path in files))

    def _populate_results_table(self, result: Dict[str, Any]):
        rows = sorted(
            result.get("samples", []),
            key=lambda row: (float(row.get("frequency_ghz", 0.0)), float(row.get("theta_scat_deg", 0.0))),
        )
        self.table_results.clearContents()
        self.table_results.setRowCount(len(rows))
        for r, row in enumerate(rows):
            freq = float(row.get("frequency_ghz", 0.0))
            elev = float(row.get("theta_scat_deg", 0.0))
            lin = float(row.get("rcs_linear", 0.0))
            db = float(row.get("rcs_db", 10.0 * math.log10(max(lin, 1e-12))))
            self.table_results.setItem(r, 0, QTableWidgetItem(f"{freq:.6g}"))
            self.table_results.setItem(r, 1, QTableWidgetItem(f"{elev:.6g}"))
            self.table_results.setItem(r, 2, QTableWidgetItem(f"{lin:.6e}"))
            self.table_results.setItem(r, 3, QTableWidgetItem(f"{db:.3f}"))

    def _plot_results(self, result: Dict[str, Any]):
        by_freq: Dict[float, List[Dict[str, Any]]] = {}
        for row in result.get("samples", []):
            freq = float(row.get("frequency_ghz", 0.0))
            by_freq.setdefault(freq, []).append(row)

        ax = self.canvas.ax
        ax.clear()
        for freq in sorted(by_freq.keys()):
            rows = sorted(by_freq[freq], key=lambda row: float(row.get("theta_scat_deg", 0.0)))
            x = [float(row.get("theta_scat_deg", 0.0)) for row in rows]
            y = [float(row.get("rcs_db", 10.0 * math.log10(max(float(row.get("rcs_linear", 1e-12)), 1e-12)))) for row in rows]
            ax.plot(x, y, linewidth=1.8, label=f"{freq:g} GHz")

        ax.set_title("Monostatic 2D RCS")
        ax.set_xlabel("Elevation (deg)")
        ax.set_ylabel("RCS (dB, relative)")
        ax.grid(True, alpha=0.3)
        if len(by_freq) <= 12:
            ax.legend(loc="best")
        self.canvas.draw_idle()
