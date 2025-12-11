#!/usr/bin/env python3
"""
FEM-ML GUI: Enhanced Main Application Window
Abaqus CAE-like interface for explicit FEM simulations
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QAction, QFileDialog, QMessageBox,
    QDockWidget, QTreeWidget, QTreeWidgetItem, QPushButton,
    QLabel, QStatusBar, QTabWidget, QProgressBar, QGroupBox,
    QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QDialog, QDialogButtonBox, QFormLayout, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QTableView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QIcon, QFont, QColor

import math
import re
import numpy as np

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def parse_node_list(text):
    nodes = []
    if not text:
        return nodes
    for token in text.replace(';', ',').split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            parts = token.split('-', 1)
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid range: {token}")
            step = 1 if end >= start else -1
            nodes.extend(range(start, end + step, step))
        else:
            try:
                nodes.append(int(token))
            except ValueError:
                raise ValueError(f"Invalid node id: {token}")
    return nodes
import textwrap

# Try importing visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not available. 3D visualization disabled.")

# Try importing plotting
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: Matplotlib not available. Plotting disabled.")

# Try importing FEM module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_script_dir, ".."))

_BC_PRESETS = {
    "Job-1 Left Clamp (Fixed)": {
        "file": os.path.join(_root_dir, "examples", "job1_gpu", "node_sets", "left_nodes.txt"),
        "bc_type": "fixed",
        "component": -1,
        "value": 0.0,
        "ramp_time": 0.0,
        "type_index": 0,
    },
    "Job-1 Right Disp (0.75mm X)": {
        "file": os.path.join(_root_dir, "examples", "job1_gpu", "node_sets", "right_nodes.txt"),
        "bc_type": "displacement",
        "component": 0,
        "value": 0.00075,
        "ramp_time": 0.3,
        "type_index": 4,
    },
    "Job-1 Top Surface": {
        "file": os.path.join(_root_dir, "examples", "job1_gpu", "node_sets", "top_nodes.txt"),
        "bc_type": "fixed",
        "component": -1,
        "value": 0.0,
        "ramp_time": 0.0,
        "type_index": 0,
    },
    "Job-1 Bottom Surface": {
        "file": os.path.join(_root_dir, "examples", "job1_gpu", "node_sets", "bottom_nodes.txt"),
        "bc_type": "fixed",
        "component": -1,
        "value": 0.0,
        "ramp_time": 0.0,
        "type_index": 0,
    },
    "Simple Cube Bottom Fixed": {
        "file": os.path.join(_root_dir, "examples", "simple_cube", "node_sets", "bottom_nodes.txt"),
        "bc_type": "fixed",
        "component": -1,
        "value": 0.0,
        "ramp_time": 0.0,
        "type_index": 0,
    },
    "Simple Cube Top Disp (0.5mm Y)": {
        "file": os.path.join(_root_dir, "examples", "simple_cube", "node_sets", "top_nodes.txt"),
        "bc_type": "displacement",
        "component": 1,
        "value": 5e-4,
        "ramp_time": 0.05,
        "type_index": 5,
    },
}

try:
    import femml
    HAS_FEMML = True
    HAS_GPU_SOLVER = hasattr(femml, "GPUExplicitSolver")
except ImportError:
    HAS_FEMML = False
    HAS_GPU_SOLVER = False
    print("Warning: femml compatibility layer not available. GUI functionality disabled.")


class MaterialDialog(QDialog):
    """Dialog for creating materials"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Material")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Material name
        self.name_edit = QLineEdit("Material-1")
        layout.addRow("Name:", self.name_edit)

        # Material type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Linear Elastic", "Neural Network"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        layout.addRow("Type:", self.type_combo)

        # Preset
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Aluminum", "Steel", "Titanium"])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        layout.addRow("Preset:", self.preset_combo)

        # Properties
        self.E_edit = QDoubleSpinBox()
        self.E_edit.setRange(1e6, 1e12)
        self.E_edit.setValue(70e9)
        self.E_edit.setDecimals(2)
        self.E_edit.setSuffix(" Pa")
        layout.addRow("Young's Modulus:", self.E_edit)

        self.nu_edit = QDoubleSpinBox()
        self.nu_edit.setRange(0.0, 0.49)
        self.nu_edit.setValue(0.33)
        self.nu_edit.setDecimals(3)
        layout.addRow("Poisson's Ratio:", self.nu_edit)

        self.rho_edit = QDoubleSpinBox()
        self.rho_edit.setRange(1.0, 50000.0)
        self.rho_edit.setValue(2700.0)
        self.rho_edit.setDecimals(1)
        self.rho_edit.setSuffix(" kg/m³")
        layout.addRow("Density:", self.rho_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setLayout(layout)

    def on_type_changed(self, index):
        is_elastic = (index == 0)
        self.E_edit.setEnabled(is_elastic)
        self.nu_edit.setEnabled(is_elastic)
        self.preset_combo.setEnabled(is_elastic)

    def on_preset_changed(self, index):
        presets = {
            1: (70e9, 0.33, 2700),   # Aluminum
            2: (210e9, 0.3, 7800),   # Steel
            3: (110e9, 0.34, 4500),  # Titanium
        }
        if index in presets:
            E, nu, rho = presets[index]
            self.E_edit.setValue(E)
            self.nu_edit.setValue(nu)
            self.rho_edit.setValue(rho)

    def get_material(self):
        """Return material parameters"""
        return {
            'name': self.name_edit.text(),
            'type': self.type_combo.currentText(),
            'E': self.E_edit.value(),
            'nu': self.nu_edit.value(),
            'rho': self.rho_edit.value()
        }


class BCDialog(QDialog):
    """Dialog for creating boundary conditions"""

    def __init__(self, node_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Boundary Condition")
        self.setModal(True)
        self.node_count = node_count
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Name
        self.name_edit = QLineEdit("BC-1")
        layout.addRow("Name:", self.name_edit)

        # Node selection
        self.nodes_edit = QLineEdit("1,2,3,4")
        self.nodes_edit.setPlaceholderText("e.g., 1,2,3,4 or 1-10")
        layout.addRow("Nodes:", self.nodes_edit)

        # Preset selector
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("None")
        for preset_name in sorted(_BC_PRESETS.keys()):
            self.preset_combo.addItem(preset_name)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        layout.addRow("Preset:", self.preset_combo)

        # BC type
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Fixed (U1=U2=U3=0)",
            "Fixed X",
            "Fixed Y",
            "Fixed Z",
            "Displacement X",
            "Displacement Y",
            "Displacement Z",
        ])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        layout.addRow("Type:", self.type_combo)

        # Displacement magnitude
        self.disp_value = QDoubleSpinBox()
        self.disp_value.setRange(-1.0, 1.0)
        self.disp_value.setDecimals(6)
        self.disp_value.setSingleStep(1e-4)
        self.disp_value.setSuffix(" m")
        self.disp_value.setValue(0.0)
        layout.addRow("Displacement:", self.disp_value)

        # Ramp time
        self.ramp_time = QDoubleSpinBox()
        self.ramp_time.setRange(0.0, 1e6)
        self.ramp_time.setDecimals(4)
        self.ramp_time.setSingleStep(0.01)
        self.ramp_time.setValue(0.0)
        self.ramp_time.setSuffix(" s")
        layout.addRow("Ramp Time:", self.ramp_time)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setLayout(layout)
        self.on_type_changed(self.type_combo.currentIndex())

    def on_type_changed(self, index):
        """Enable displacement inputs only for displacement BCs."""
        is_displacement = index >= 4
        self.disp_value.setEnabled(is_displacement)
        self.ramp_time.setEnabled(is_displacement)

    def on_preset_changed(self, index):
        """Apply Job-1 presets for node selections."""
        if index <= 0:
            return
        preset_name = self.preset_combo.currentText()
        preset = _BC_PRESETS.get(preset_name)
        if not preset:
            return
        file_path = preset.get("file")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                nodes_str = f.read().strip()
            self.nodes_edit.setText(nodes_str)
        except OSError as osex:
            QMessageBox.warning(self, "Preset Error", f"Failed to read node set:\n{file_path}\n{osex}")
            return

        # Apply type and component suggestions
        self.type_combo.setCurrentIndex(preset.get("type_index", 0))
        if preset.get("bc_type") == "displacement":
            self.disp_value.setValue(preset.get("value", 0.0))
            self.ramp_time.setValue(preset.get("ramp_time", 0.0))
        else:
            self.disp_value.setValue(0.0)
            self.ramp_time.setValue(0.0)

    def get_bc(self):
        """Return BC parameters"""
        nodes_text = self.nodes_edit.text()
        nodes = parse_node_list(nodes_text)

        component_map = {
            0: -1,  # Fixed all
            1: 0,   # Fixed X
            2: 1,   # Fixed Y
            3: 2,   # Fixed Z
            4: 0,   # Disp X
            5: 1,   # Disp Y
            6: 2,   # Disp Z
        }

        index = self.type_combo.currentIndex()
        bc_type = "displacement" if index >= 4 else "fixed"

        return {
            'name': self.name_edit.text(),
            'nodes': nodes,
            'component': component_map.get(index, -1),
            'bc_type': bc_type,
            'value': self.disp_value.value(),
            'ramp_time': self.ramp_time.value()
        }


class LoadDialog(QDialog):
    """Dialog for creating loads"""

    def __init__(self, node_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Load")
        self.setModal(True)
        self.node_count = node_count
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Name
        self.name_edit = QLineEdit("Load-1")
        layout.addRow("Name:", self.name_edit)

        # Node selection
        self.nodes_edit = QLineEdit("8")
        layout.addRow("Nodes:", self.nodes_edit)

        # Direction
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["X", "Y", "Z"])
        self.dir_combo.setCurrentIndex(1)  # Y default
        layout.addRow("Direction:", self.dir_combo)

        # Magnitude
        self.mag_edit = QDoubleSpinBox()
        self.mag_edit.setRange(-1e9, 1e9)
        self.mag_edit.setValue(5e5)
        self.mag_edit.setDecimals(0)
        self.mag_edit.setSuffix(" N")
        layout.addRow("Magnitude:", self.mag_edit)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setLayout(layout)

    def get_load(self):
        """Return load parameters"""
        nodes_text = self.nodes_edit.text()
        nodes = parse_node_list(nodes_text)

        return {
            'name': self.name_edit.text(),
            'nodes': nodes,
            'component': self.dir_combo.currentIndex(),
            'value': self.mag_edit.value()
        }


class SolverParamsDialog(QDialog):
    """Dialog for solver parameters"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Solver Parameters")
        self.setModal(True)
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Time step
        self.dt_edit = QDoubleSpinBox()
        self.dt_edit.setRange(1e-12, 1e-3)
        self.dt_edit.setValue(5e-8)
        self.dt_edit.setDecimals(10)
        self.dt_edit.setSuffix(" s")
        layout.addRow("Time Step:", self.dt_edit)

        # Auto time step
        self.auto_dt_check = QCheckBox("Auto (compute from stability)")
        self.auto_dt_check.setChecked(True)
        layout.addRow("", self.auto_dt_check)

        # Number of steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1000000)
        self.steps_spin.setValue(2000)
        layout.addRow("Number of Steps:", self.steps_spin)

        # Damping
        self.damp_edit = QDoubleSpinBox()
        self.damp_edit.setRange(0.0, 1.0)
        self.damp_edit.setValue(0.0)
        self.damp_edit.setDecimals(3)
        layout.addRow("Damping:", self.damp_edit)

        # Output interval
        self.output_spin = QSpinBox()
        self.output_spin.setRange(1, 10000)
        self.output_spin.setValue(50)
        layout.addRow("Output Interval:", self.output_spin)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setLayout(layout)

    def get_params(self):
        """Return solver parameters"""
        return {
            'dt': self.dt_edit.value(),
            'auto_dt': self.auto_dt_check.isChecked(),
            'steps': self.steps_spin.value(),
            'damping': self.damp_edit.value(),
            'output_interval': self.output_spin.value()
        }


class ModelTree(QTreeWidget):
    """Enhanced model tree"""

    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Model Tree")
        self.setMinimumWidth(250)
        self.setup_tree()

    def setup_tree(self):
        # Parts
        self.parts_root = QTreeWidgetItem(self, ["Parts"])
        font = self.parts_root.font(0)
        font.setBold(True)
        self.parts_root.setFont(0, font)

        # Materials
        self.materials_root = QTreeWidgetItem(self, ["Materials"])
        self.materials_root.setFont(0, font)

        # Steps
        self.steps_root = QTreeWidgetItem(self, ["Steps"])
        self.steps_root.setFont(0, font)
        self.step1 = QTreeWidgetItem(self.steps_root, ["Step-1 (Explicit)"])

        # Loads
        self.loads_root = QTreeWidgetItem(self.step1, ["Loads"])

        # Boundary Conditions
        self.bcs_root = QTreeWidgetItem(self.step1, ["Boundary Conditions"])

        # Mesh
        self.mesh_root = QTreeWidgetItem(self, ["Mesh"])
        self.mesh_root.setFont(0, font)

        # Jobs
        self.jobs_root = QTreeWidgetItem(self, ["Jobs"])
        self.jobs_root.setFont(0, font)

        # Results
        self.results_root = QTreeWidgetItem(self, ["Results"])
        self.results_root.setFont(0, font)

        self.expandAll()

class JobMonitor(QWidget):
    """Enhanced job monitoring panel"""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Job Monitor")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title.setFont(title_font)
        layout.addWidget(title)

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Status: Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QFormLayout()

        self.step_label = QLabel("0 / 0")
        self.time_label = QLabel("0.0 s")

        progress_layout.addRow("Step:", self.step_label)
        progress_layout.addRow("Time:", self.time_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Energy group
        energy_group = QGroupBox("Energy")
        energy_layout = QFormLayout()

        self.ke_label = QLabel("0.0 J")
        self.se_label = QLabel("0.0 J")
        self.te_label = QLabel("0.0 J")

        energy_layout.addRow("Kinetic:", self.ke_label)
        energy_layout.addRow("Strain:", self.se_label)
        energy_layout.addRow("Total:", self.te_label)
        energy_group.setLayout(energy_layout)
        layout.addWidget(energy_group)

        layout.addStretch()
        self.setLayout(layout)

    def update_status(self, step, total_steps, time, ke, se=None, te=None):
        progress = int(100.0 * step / total_steps) if total_steps > 0 else 0
        self.status_label.setText(f"Status: Running (Step {step})")
        self.progress_bar.setValue(progress)
        self.step_label.setText(f"{step} / {total_steps}")
        self.time_label.setText(f"{time:.6e} s")
        self.ke_label.setText(f"{ke:.6e} J")
        if se is not None:
            self.se_label.setText(f"{se:.6e} J")
        if te is not None:
            self.te_label.setText(f"{te:.6e} J")

    def reset(self):
        self.status_label.setText("Status: Ready")
        self.progress_bar.setValue(0)
        self.step_label.setText("0 / 0")
        self.time_label.setText("0.0 s")
        self.ke_label.setText("0.0 J")
        self.se_label.setText("0.0 J")
        self.te_label.setText("0.0 J")


class SimpleTableModel(QAbstractTableModel):
    """Generic table model for displaying solver data."""

    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.headers = headers
        self._rows = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        value = self._rows[index.row()][index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, float):
                return f"{value:.6e}"
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, (int, float)):
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ForegroundRole and isinstance(value, float) and abs(value) < 1e-12:
            return QColor("#888888")
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
        if orientation == Qt.Horizontal and section < len(self.headers):
            return self.headers[section]
        if orientation == Qt.Vertical:
            return section + 1
        return QVariant()

    def update(self, rows):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def clear(self):
        self.update([])


class ResultsDock(QDockWidget):
    """Dock widget to display nodal and elemental results."""

    def __init__(self, parent=None):
        super().__init__("Results", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.node_model = SimpleTableModel(
            ["Node", "X", "Y", "Z", "Ux", "Uy", "Uz"], self)
        self.node_table = QTableView()
        self.node_table.setModel(self.node_model)
        self.node_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.node_table.setAlternatingRowColors(True)
        self.node_table.horizontalHeader().setStretchLastSection(True)
        self.node_table.verticalHeader().setVisible(False)
        self.tabs.addTab(self.node_table, "Nodes")

        self.element_model = SimpleTableModel(
            ["Element", "Type", "Volume",
             "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz",
             "Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"], self)
        self.element_table = QTableView()
        self.element_table.setModel(self.element_model)
        self.element_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.element_table.setAlternatingRowColors(True)
        self.element_table.horizontalHeader().setStretchLastSection(True)
        self.element_table.verticalHeader().setVisible(False)
        self.tabs.addTab(self.element_table, "Elements")

        layout.addWidget(self.tabs)
        container.setLayout(layout)
        self.setWidget(container)
        self.hide()

    def update_node_results(self, node_results):
        rows = []
        for res in node_results:
            coords = list(res.coords) if hasattr(res, "coords") else res["coords"]
            disp = list(res.displacement) if hasattr(res, "displacement") else res["displacement"]
            rows.append([
                res.id if hasattr(res, "id") else res["id"],
                coords[0], coords[1], coords[2],
                disp[0], disp[1], disp[2]
            ])
        self.node_model.update(rows)
        if rows:
            self.show()

    def update_element_results(self, element_results):
        rows = []
        def _get(res, attr: str, key: str, default=None):
            if hasattr(res, attr):
                return getattr(res, attr)
            if isinstance(res, dict) and key in res:
                return res[key]
            return default

        for res in element_results:
            stress = _get(res, "stress", "stress", [0.0] * 6)
            strain = _get(res, "strain", "strain", [0.0] * 6)
            if stress is None:
                stress = [0.0] * 6
            if strain is None:
                strain = [0.0] * 6

            stress = list(stress)
            strain = list(strain)
            # Pad to length 6 in case shorter sequences arrive
            stress = (stress + [0.0] * 6)[:6]
            strain = (strain + [0.0] * 6)[:6]

            rows.append([
                _get(res, "id", "id", ""),
                _get(res, "type", "type", ""),
                _get(res, "volume", "volume", 0.0),
                *stress,
                *strain
            ])
        self.element_model.update(rows)
        if rows:
            self.show()

    def clear_results(self):
        self.node_model.clear()
        self.element_model.clear()
        self.hide()


class Viewer3D(QWidget if not HAS_PYVISTA else QtInteractor):
    """Enhanced 3D visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)
        if HAS_PYVISTA:
            self.mesh_actor = None
            self.mesh_data = None
            self.base_points = None
            self._text_actor = None
        else:
            # Fallback UI
            layout = QVBoxLayout()
            label = QLabel("3D Visualization\n(PyVista not available)")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.setLayout(layout)

    def _remove_mesh_actor(self, actor=None):
        """Remove a mesh actor without clearing the entire scene."""
        if not HAS_PYVISTA:
            return
        target = actor if actor is not None else self.mesh_actor
        if target is None:
            return
        try:
            self.remove_actor(target, reset_camera=False, render=False)
        except Exception:
            try:
                self.renderer.RemoveActor(target)
            except Exception:
                pass
        if actor is None or target is self.mesh_actor:
            if target is self.mesh_actor:
                self.mesh_actor = None

    def _update_title_text(self, title: str = ""):
        """Refresh the scalar-bar title text actor."""
        if not HAS_PYVISTA:
            return
        if self._text_actor is not None:
            try:
                self.remove_actor(self._text_actor, reset_camera=False, render=False)
            except Exception:
                try:
                    self.renderer.RemoveActor2D(self._text_actor)
                except Exception:
                    pass
            self._text_actor = None
        if title:
            try:
                self._text_actor = self.add_text(title, font_size=10, name="_field_title")
            except Exception:
                self._text_actor = None

    def _replace_mesh_actor(self, frame, plot_kwargs, title=""):
        """Add a new mesh actor and remove the previous one only after success."""
        if not HAS_PYVISTA:
            return False
        old_actor = self.mesh_actor
        try:
            new_actor = self.add_mesh(frame, **plot_kwargs)
        except Exception:
            return False
        self.mesh_actor = new_actor
        self._update_title_text(title)
        if old_actor is not None:
            self._remove_mesh_actor(old_actor)
        return True

    def load_mesh(self, nodes, elements):
        """Load mesh for visualization"""
        if not HAS_PYVISTA:
            return

        points = np.array([[n.coords[0], n.coords[1], n.coords[2]] for n in nodes])

        cell_records = []
        cell_types = []
        for elem in elements:
            elem_type = getattr(elem, "type", "")
            node_list = list(getattr(elem, "nodes", []))
            vtk_type = None

            if elem_type in {"C3D8", "C3D8R"} and len(node_list) == 8:
                vtk_type = pv.CellType.HEXAHEDRON
            elif elem_type == "C3D4" and len(node_list) == 4:
                vtk_type = pv.CellType.TETRA

            if vtk_type is None:
                continue

            node_ids = [nid - 1 for nid in node_list]
            cell_records.extend([len(node_ids), *node_ids])
            cell_types.append(vtk_type)

        if not cell_types:
            raise ValueError("No supported elements available for visualization")

        cells_array = np.array(cell_records, dtype=np.int64)
        cell_types_array = np.array(cell_types, dtype=np.uint8)
        self.mesh_data = pv.UnstructuredGrid(cells_array, cell_types_array, points)
        self.base_points = self.mesh_data.points.copy()

        plot_kwargs = {
            'show_edges': True,
            'color': 'lightblue',
            'opacity': 0.9,
            'edge_color': 'black'
        }
        if not self._replace_mesh_actor(self.mesh_data, plot_kwargs, title=""):
            raise RuntimeError("Failed to initialize PyVista mesh actor")

        self.reset_camera()
        self.update()

    def update_field(self, displacements=None, scalars=None, title="", scale=1.0,
                     cmap="viridis", show_scalar_bar=True):
        """Update visualization with optional displacement field and scalars."""
        if not HAS_PYVISTA or self.mesh_data is None:
            return

        frame = self.mesh_data.copy()
        if self.base_points is not None:
            frame.points = self.base_points.copy()
        if displacements is not None and self.base_points is not None:
            disp_array = np.asarray(displacements, dtype=float)
            disp_array = np.nan_to_num(disp_array, nan=0.0, posinf=1e6, neginf=-1e6)
            disp_array = disp_array * scale
            if disp_array.shape == self.base_points.shape:
                frame.points = self.base_points + disp_array

        plot_kwargs = {
            'show_edges': True,
        }

        if scalars is not None:
            scalars = np.asarray(scalars, dtype=float)
            scalars = np.nan_to_num(scalars, nan=0.0, posinf=1e12, neginf=-1e12)
            if scalars.shape[0] == frame.n_points:
                plot_kwargs.update({
                    'scalars': scalars,
                    'cmap': cmap,
                    'show_scalar_bar': show_scalar_bar,
                    'scalar_bar_args': {'title': title or '', 'vertical': True},
                })
            else:
                frame.clear_data()
                frame.cell_data["field"] = scalars
                plot_kwargs.update({
                    'scalars': scalars,
                    'cmap': cmap,
                    'show_scalar_bar': show_scalar_bar,
                    'scalar_bar_args': {'title': title or '', 'vertical': True},
                })
        else:
            plot_kwargs.update({
                'color': 'lightblue',
                'opacity': 0.9,
                'edge_color': 'black',
                'show_scalar_bar': False,
            })

        if self._replace_mesh_actor(frame, plot_kwargs, title=title):
            self.update()

    def update_displacements(self, displacements, scale=1000.0):
        disp_mag = np.linalg.norm([[d[0], d[1], d[2]] for d in displacements], axis=1)
        self.update_field(displacements=displacements, scalars=disp_mag,
                          title="Displacement (m)", scale=scale, cmap="jet")

    def save_screenshot(self, path, transparent=False):
        """Save a screenshot of the current PyVista view."""
        if not HAS_PYVISTA or self.mesh_actor is None:
            return False
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.screenshot(filename=str(path), transparent_background=transparent)
            return True
        except Exception:
            return False


class MainWindow(QMainWindow):
    """Enhanced main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM-ML: Explicit FEM Solver")
        self.setGeometry(100, 100, 1600, 1000)

        # Data
        self.mesh = None
        self.solver = None
        self.materials = {}
        self.bcs = []
        self.loads = []
        self.deformation_scale = 1.0
        self.use_gpu = HAS_GPU_SOLVER
        self.show_scalar_bar = True
        self.is_solving = False
        self.solver_params = None
        self.mesh_nodes_cache = None
        self.mesh_coords_np = None
        self.mesh_conn_np = None
        self.mesh_node_ids = None
        self.mesh_element_ids = None
        self.node_index_map = {}
        self.element_index_map = {}
        self.display_field = "Von Mises Stress"
        self.last_displacements = None
        self.last_stress_components = None
        self.last_vm = None
        self.enable_live_view = True
        self.export_frames = False
        self.frame_export_interval = 50
        self._frame_export_warned = False

        self.setup_ui()
        self.create_menus()
        self.create_toolbars()
        self.create_status_bar()

        # Show welcome message
        self.show_welcome()

    def setup_ui(self):
        # Central widget
        self.viewer = Viewer3D(self)
        self.setCentralWidget(self.viewer)

        # Left dock: Model tree
        left_dock = QDockWidget("Model", self)
        left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.model_tree = ModelTree()
        left_dock.setWidget(self.model_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # Right dock: Properties
        right_dock = QDockWidget("Properties", self)
        right_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.property_panel = QTextEdit()
        self.property_panel.setReadOnly(True)
        self.property_panel.setText("Select an item to view properties")
        right_dock.setWidget(self.property_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # Bottom dock: Job monitor
        bottom_dock = QDockWidget("Job Monitor", self)
        bottom_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.job_monitor = JobMonitor()
        bottom_dock.setWidget(self.job_monitor)
        self.addDockWidget(Qt.BottomDockWidgetArea, bottom_dock)
        self.job_dock = bottom_dock

        # Results dock (tabbed with job monitor)
        self.results_dock = ResultsDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.results_dock)
        self.tabifyDockWidget(bottom_dock, self.results_dock)
        self.results_dock.hide()

    def create_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        import_action = QAction("&Import Mesh...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_mesh)
        file_menu.addAction(import_action)

        export_action = QAction("&Export Results...", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Model menu
        model_menu = menubar.addMenu("&Model")

        material_action = QAction("Create &Material...", self)
        material_action.triggered.connect(self.create_material)
        model_menu.addAction(material_action)

        bc_action = QAction("Create &Boundary Condition...", self)
        bc_action.triggered.connect(self.create_bc)
        model_menu.addAction(bc_action)

        load_action = QAction("Create &Load...", self)
        load_action.triggered.connect(self.create_load)
        model_menu.addAction(load_action)

        # Job menu
        job_menu = menubar.addMenu("&Job")

        params_action = QAction("Solver &Parameters...", self)
        params_action.triggered.connect(self.edit_solver_params)
        job_menu.addAction(params_action)

        job_menu.addSeparator()

        submit_action = QAction("&Submit Job", self)
        submit_action.setShortcut("F5")
        submit_action.triggered.connect(self.submit_job)
        job_menu.addAction(submit_action)

        stop_action = QAction("S&top Job", self)
        stop_action.triggered.connect(self.stop_job)
        job_menu.addAction(stop_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_view = QAction("&Reset View", self)
        reset_view.triggered.connect(lambda: hasattr(self.viewer, 'reset_camera') and self.viewer.reset_camera())
        view_menu.addAction(reset_view)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        howto_action = QAction("&How to Run", self)
        howto_action.triggered.connect(self.show_howto)
        help_menu.addAction(howto_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbars(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        import_btn = QAction("Import Mesh", self)
        import_btn.triggered.connect(self.import_mesh)
        toolbar.addAction(import_btn)

        toolbar.addSeparator()

        mat_btn = QAction("Create Material", self)
        mat_btn.triggered.connect(self.create_material)
        toolbar.addAction(mat_btn)

        bc_btn = QAction("Boundary Condition", self)
        bc_btn.triggered.connect(self.create_bc)
        toolbar.addAction(bc_btn)

        load_btn = QAction("Load", self)
        load_btn.triggered.connect(self.create_load)
        toolbar.addAction(load_btn)

        toolbar.addSeparator()

        run_btn = QAction("Run", self)
        run_btn.triggered.connect(self.submit_job)
        toolbar.addAction(run_btn)

        stop_btn = QAction("Stop", self)
        stop_btn.triggered.connect(self.stop_job)
        toolbar.addAction(stop_btn)

        toolbar.addSeparator()

        scale_label = QLabel("Deformation Scale:")
        toolbar.addWidget(scale_label)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 1000.0)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(self.deformation_scale)
        self.scale_spin.editingFinished.connect(self.on_scale_changed)
        toolbar.addWidget(self.scale_spin)

        self.gpu_checkbox = QCheckBox("Use GPU")
        self.gpu_checkbox.setChecked(self.use_gpu)
        self.gpu_checkbox.setEnabled(HAS_GPU_SOLVER)
        if not HAS_GPU_SOLVER:
            self.gpu_checkbox.setToolTip("GPU solver not available. Rebuild with CUDA support to enable.")
        self.gpu_checkbox.toggled.connect(self.on_gpu_toggled)
        toolbar.addWidget(self.gpu_checkbox)

        self.legend_checkbox = QCheckBox("Show Legend")
        self.legend_checkbox.setChecked(self.show_scalar_bar)
        self.legend_checkbox.toggled.connect(self.on_legend_toggled)
        toolbar.addWidget(self.legend_checkbox)
        toolbar.addWidget(QLabel(" Field: "))
        self.field_combo = QComboBox()
        self.field_combo.addItems([
            "Von Mises Stress",
            "Displacement Magnitude",
            "Sxx",
            "Syy",
            "Szz",
            "Sxy",
            "Sxz",
            "Syz",
        ])
        self.field_combo.setCurrentText(self.display_field)
        self.field_combo.currentTextChanged.connect(self.on_field_changed)
        toolbar.addWidget(self.field_combo)

        self.live_view_checkbox = QCheckBox("Live View")
        self.live_view_checkbox.setChecked(self.enable_live_view)
        self.live_view_checkbox.toggled.connect(self.on_live_view_toggled)
        toolbar.addWidget(self.live_view_checkbox)

        self.frame_export_checkbox = QCheckBox("Export Frames")
        self.frame_export_checkbox.setChecked(self.export_frames)
        self.frame_export_checkbox.toggled.connect(self.on_frame_export_toggled)
        toolbar.addWidget(self.frame_export_checkbox)

        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 1000)
        self.frame_interval_spin.setValue(self.frame_export_interval)
        self.frame_interval_spin.setEnabled(self.export_frames)
        self.frame_interval_spin.valueChanged.connect(self.on_frame_interval_changed)
        toolbar.addWidget(QLabel(" every "))
        toolbar.addWidget(self.frame_interval_spin)
        toolbar.addWidget(QLabel(" steps"))

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Import a mesh to begin")

    def show_welcome(self):
        """Show welcome message"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Welcome to FEM-ML")
        msg.setText("Welcome to FEM-ML!")
        msg.setInformativeText(textwrap.dedent("""\
            Quick Start:
            1. File -> Import Mesh (try examples/simple_cube/cube.inp)
            2. Model -> Create Material
            3. Model -> Create Boundary Condition
            4. Model -> Create Load
            5. Job -> Submit Job

            Need help? Help -> How to Run
        """))
        msg.exec_()

    def import_mesh(self):
        """Import mesh from file"""
        if not HAS_FEMML:
            QMessageBox.critical(self, "Error",
                "femml module not available.\n"
                "Please build the project first:\n"
                "  cd FEM-ML\n"
                "  .\\build.ps1")
            return

        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Mesh",
            os.path.join(_root_dir, "examples", "simple_cube"),
            "Abaqus Input Files (*.inp);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.status_bar.showMessage(f"Importing {os.path.basename(filename)}...")
            QApplication.processEvents()

            # Import
            importer = femml.AbaqusImporter()
            self.mesh = importer.import_mesh(filename)
            try:
                coords_np, conn_np, node_ids, element_ids = self.mesh.to_numpy()
                self.mesh_coords_np = coords_np
                self.mesh_conn_np = conn_np
                self.mesh_node_ids = node_ids
                self.mesh_element_ids = element_ids
                self.node_index_map = {nid: idx for idx, nid in enumerate(node_ids)}
                self.element_index_map = {eid: idx for idx, eid in enumerate(element_ids)}
            except Exception:
                self.mesh_coords_np = None
                self.mesh_conn_np = None
                self.mesh_node_ids = None
                self.mesh_element_ids = None
                self.node_index_map = {}
                self.element_index_map = {}

            # Get nodes and elements
            nodes = []
            for i in range(1, self.mesh.get_num_nodes() + 1):
                try:
                    nodes.append(self.mesh.get_node(i))
                except:
                    pass

            elements = []
            for i in range(1, self.mesh.get_num_elements() + 1):
                try:
                    elements.append(self.mesh.get_element(i))
                except:
                    pass

            # Update tree
            self.model_tree.mesh_root.takeChildren()
            QTreeWidgetItem(self.model_tree.mesh_root, [f"Nodes: {len(nodes)}"])
            QTreeWidgetItem(self.model_tree.mesh_root, [f"Elements: {len(elements)}"])
            self.model_tree.expandAll()

            # Visualize
            if HAS_PYVISTA and hasattr(self.viewer, 'load_mesh'):
                self.viewer.load_mesh(nodes, elements)
                self.mesh_nodes_cache = nodes

                self.status_bar.showMessage(
                    f"Loaded {len(nodes)} nodes, {len(elements)} elements"
                )

                QMessageBox.information(self, "Success",
                    f"Imported mesh successfully:\n"
                    f"  Nodes: {len(nodes)}\n"
                    f"  Elements: {len(elements)}")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import mesh:\n{str(e)}")
            self.status_bar.showMessage("Import failed")
            return

        self._apply_metadata(Path(filename))

    def _apply_metadata(self, inp_path: Path) -> None:
        """Load optional metadata (BCs, materials, solver params) for exported meshes."""
        meta_path = Path(str(inp_path) + ".meta.json")
        if not meta_path.exists():
            return
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception as exc:
            self.status_bar.showMessage(f"Metadata load failed: {exc}")
            return

        # Materials
        self.materials.clear()
        self.model_tree.materials_root.takeChildren()
        material_info = meta.get("material")
        if material_info and HAS_FEMML:
            material = femml.LinearElastic(
                material_info.get("name", "Material-1"),
                material_info.get("E", 70e9),
                material_info.get("nu", 0.33),
                material_info.get("rho", 2700.0),
            )
            self.materials[material.name] = material
            QTreeWidgetItem(
                self.model_tree.materials_root,
                [f"{material.name} (Linear Elastic)"],
            )

        # Boundary conditions
        self.bcs.clear()
        self.model_tree.bcs_root.takeChildren()
        for bc in meta.get("boundary_conditions", []):
            entry = {
                "name": bc.get("name", "BC"),
                "nodes": [int(n) for n in bc.get("nodes", [])],
                "component": int(bc.get("component", -1)),
                "bc_type": bc.get("type", "fixed"),
                "value": bc.get("value", 0.0),
                "ramp_time": bc.get("ramp_time", 0.0),
            }
            self.bcs.append(entry)
            if entry["bc_type"] == "displacement":
                dir_map = {0: "X", 1: "Y", 2: "Z", -1: "All"}
                direction = dir_map.get(entry["component"], "?")
                magnitude_mm = entry["value"] * 1e3
                desc = f"{entry['name']} (Disp {magnitude_mm:.3f} mm {direction})"
            else:
                desc = f"{entry['name']} ({len(entry['nodes'])} nodes fixed)"
            QTreeWidgetItem(self.model_tree.bcs_root, [desc])

        # Loads (none by default)
        self.loads.clear()
        self.model_tree.loads_root.takeChildren()

        # Solver params
        solver_meta = meta.get("solver_params")
        if solver_meta:
            self.solver_params = solver_meta

        self.model_tree.expandAll()
        self.status_bar.showMessage(
            f"Metadata applied from {meta_path.name} (BCs/material/solver parameters)"
        )

    def create_material(self):
        """Create material"""
        if not HAS_FEMML:
            QMessageBox.warning(self, "Error", "femml module not available")
            return

        dialog = MaterialDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_material()

            # Create material
            if params['type'] == "Linear Elastic":
                material = femml.LinearElastic(
                    params['name'],
                    params['E'],
                    params['nu'],
                    params['rho']
                )
            else:
                material = femml.NeuralMaterial(params['name'], params['rho'])

            self.materials[params['name']] = material

            # Add to tree
            QTreeWidgetItem(self.model_tree.materials_root,
                [f"{params['name']} ({params['type']})"])
            self.model_tree.expandAll()

            self.status_bar.showMessage(f"Created material: {params['name']}")

    def create_bc(self):
        """Create boundary condition"""
        if self.mesh is None:
            QMessageBox.warning(self, "Error", "Please import a mesh first")
            return

        dialog = BCDialog(self.mesh.get_num_nodes(), self)
        if dialog.exec_() == QDialog.Accepted:
            bc_data = dialog.get_bc()
            self.bcs.append(bc_data)

            # Add to tree
            if bc_data['bc_type'] == "displacement":
                dir_map = {0: "X", 1: "Y", 2: "Z", -1: "All"}
                direction = dir_map.get(bc_data['component'], "?")
                magnitude_mm = bc_data['value'] * 1e3
                desc = f"{bc_data['name']} (Disp {magnitude_mm:.3f} mm {direction})"
            else:
                desc = f"{bc_data['name']} ({len(bc_data['nodes'])} nodes fixed)"

            QTreeWidgetItem(self.model_tree.bcs_root, [desc])
            self.model_tree.expandAll()

            self.status_bar.showMessage(f"Created BC: {bc_data['name']}")

    def create_load(self):
        """Create load"""
        if self.mesh is None:
            QMessageBox.warning(self, "Error", "Please import a mesh first")
            return

        dialog = LoadDialog(self.mesh.get_num_nodes(), self)
        if dialog.exec_() == QDialog.Accepted:
            load_data = dialog.get_load()
            self.loads.append(load_data)

            # Add to tree
            dir_name = ['X', 'Y', 'Z'][load_data['component']]
            QTreeWidgetItem(self.model_tree.loads_root,
                [f"{load_data['name']} ({load_data['value']:.0e} N, {dir_name})"])
            self.model_tree.expandAll()

            self.status_bar.showMessage(f"Created load: {load_data['name']}")

    def edit_solver_params(self):
        """Edit solver parameters"""
        dialog = SolverParamsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.solver_params = dialog.get_params()
            self.status_bar.showMessage("Solver parameters updated")

    def submit_job(self):
        """Submit analysis job"""
        if not HAS_FEMML:
            QMessageBox.warning(self, "Error", "femml module not available")
            return

        if self.mesh is None:
            QMessageBox.warning(self, "Error", "No mesh loaded")
            return

        self.job_monitor.reset()
        if hasattr(self, "results_dock"):
            self.results_dock.clear_results()
        self.last_displacements = None
        self.last_stress_components = None
        self.last_vm = None

        if not self.materials:
            QMessageBox.warning(self, "Error", "No materials defined")
            return

        if self.is_solving:
            QMessageBox.warning(self, "Error", "Job already running")
            return

        try:
            self.is_solving = True

            # Create solver
            if self.use_gpu and HAS_GPU_SOLVER:
                self.status_bar.showMessage("Starting analysis on GPU...")
                self.solver = femml.GPUExplicitSolver(self.mesh)
            else:
                if self.use_gpu and not HAS_GPU_SOLVER:
                    QMessageBox.warning(self, "GPU Solver Unavailable",
                                        "GPU backend not available. Falling back to CPU solver.")
                self.status_bar.showMessage("Starting analysis on CPU...")
                self.solver = femml.ExplicitSolver(self.mesh)

            # Assign material (use first one)
            mat_name = list(self.materials.keys())[0]
            self.solver.set_material(self.materials[mat_name])

            # Add BCs
            for bc_data in self.bcs:
                bc = femml.BoundaryCondition()
                if bc_data.get('bc_type') == "displacement":
                    bc.type = femml.BCType.DISPLACEMENT
                    bc.value = bc_data.get('value', 0.0)
                    bc.ramp_time = bc_data.get('ramp_time', 0.0)
                else:
                    bc.type = femml.BCType.FIXED
                    bc.value = 0.0
                    bc.ramp_time = 0.0
                bc.nodes = bc_data['nodes']
                bc.component = bc_data['component']
                self.solver.add_boundary_condition(bc)

            # Add loads
            for load_data in self.loads:
                for node_id in load_data['nodes']:
                    load = femml.Load()
                    load.type = femml.LoadType.FORCE
                    load.nodes = [node_id]
                    load.component = load_data['component']
                    load.value = load_data['value']
                    self.solver.add_load(load)

            # Set parameters
            if self.solver_params is None:
                self.solver_params = {
                    'dt': 5e-8,
                    'auto_dt': True,
                    'steps': 2000,
                    'damping': 0.0,
                    'output_interval': 50
                }

            params = femml.SolverParams()
            params.time_step = self.solver_params['dt']
            params.num_steps = self.solver_params['steps']
            params.damping = self.solver_params['damping']
            params.output_interval = self.solver_params['output_interval']
            params.auto_time_step = self.solver_params['auto_dt']
            self.solver.set_parameters(params)

            # Initialize
            self.solver.initialize()

            # Start solving
            self.total_steps = self.solver_params['steps']
            self.run_solver_step()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start job:\n{str(e)}")
            self.is_solving = False
            self.status_bar.showMessage("Job failed")

    def run_solver_step(self):
        """Run one solver step"""
        if not self.is_solving or self.solver.is_finished():
            # Done
            self.is_solving = False
            self.status_bar.showMessage("Analysis complete!")

            # Final status update
            step = self.solver.get_current_step()
            time = self.solver.get_current_time()
            ke = self.solver.get_kinetic_energy()
            se = self.solver.get_strain_energy()
            te = self.solver.get_total_energy()
            self.job_monitor.update_status(step, self.total_steps, time, ke, se, te)

            # Update final visualization
            self.last_displacements = self.solver.get_displacements()
            stress_components = self.compute_nodal_stress_components()
            if stress_components is not None:
                self.last_stress_components = stress_components
                self.last_vm = self._von_mises_array(stress_components)
            else:
                self.last_stress_components = None
                self.last_vm = None
            if self.export_frames:
                self.export_frame_snapshot(step, time)
            if self.enable_live_view:
                self.update_visualization_from_cache()

            self.update_results_views()
            self.auto_export_plots()
            QMessageBox.information(self, "Complete", "Analysis completed successfully!")
            return

        # Run one step
        self.solver.step()

        # Update monitor
        step = self.solver.get_current_step()
        time = self.solver.get_current_time()
        ke = self.solver.get_kinetic_energy()
        se = self.solver.get_strain_energy()
        te = self.solver.get_total_energy()
        self.job_monitor.update_status(step, self.total_steps, time, ke, se, te)

        # Update visualization/export every 10 steps
        if step % 10 == 0:
            disps = self.solver.get_displacements()
            self.last_displacements = disps
            stress_components = self.compute_nodal_stress_components()
            if stress_components is not None:
                self.last_stress_components = stress_components
                self.last_vm = self._von_mises_array(stress_components)
            else:
                self.last_stress_components = None
                self.last_vm = None

            if (
                self.export_frames
                and self.frame_export_interval > 0
                and step % self.frame_export_interval == 0
            ):
                self.export_frame_snapshot(step, time)

            if self.enable_live_view:
                self.update_visualization_from_cache()

        # Schedule next step
        QTimer.singleShot(1, self.run_solver_step)

    def update_results_views(self):
        """Refresh docked result tables after a run."""
        if self.solver is None or not hasattr(self, "results_dock"):
            return

        node_results = []
        element_results = []

        if hasattr(self.solver, "get_node_results"):
            try:
                node_results = self.solver.get_node_results()
            except Exception:
                node_results = []

        if hasattr(self.solver, "get_element_results"):
            try:
                element_results = self.solver.get_element_results()
            except Exception:
                element_results = []

        if node_results:
            self.results_dock.update_node_results(node_results)
        if element_results:
            self.results_dock.update_element_results(element_results)
        if node_results or element_results:
            self.results_dock.show()

    def stop_job(self):
        """Stop running job"""
        if self.is_solving:
            self.is_solving = False
            self.status_bar.showMessage("Job stopped by user")
            self.job_monitor.reset()

    def export_results(self):
        """Export results"""
        if self.solver is None:
            QMessageBox.warning(self, "Error", "No results to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            self.solver.write_results(filename)
            self.status_bar.showMessage(f"Results exported to {filename}")
            QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")

    def auto_export_plots(self):
        """Save viewer screenshot and summary plots after a run."""
        if self.last_displacements is None:
            return
        export_dir = Path(_root_dir) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []

        screenshot_path = export_dir / f"viewer_stress_{timestamp}.png"
        if self._export_pyvista_snapshot(screenshot_path):
            saved.append(screenshot_path)
        elif HAS_PYVISTA and hasattr(self.viewer, "save_screenshot"):
            if self.viewer.save_screenshot(screenshot_path):
                saved.append(screenshot_path)

        contour_path = export_dir / f"stress_summary_{timestamp}.png"
        if self._export_matplotlib_contours(contour_path):
            saved.append(contour_path)

        if saved:
            joined = ", ".join(str(path) for path in saved)
            self.status_bar.showMessage(f"Plots exported to {joined}")

    def _export_matplotlib_contours(self, output_path: Path) -> bool:
        """Export von Mises and displacement scatter plots via matplotlib."""
        if not HAS_MATPLOTLIB:
            return False
        if self.mesh_coords_np is None or self.mesh_conn_np is None:
            return False
        if self.solver is None or self.last_displacements is None:
            return False
        try:
            elem_results = self.solver.get_element_results()
        except Exception:
            return False
        if not elem_results:
            return False

        if not self.element_index_map and self.mesh_element_ids is not None:
            self.element_index_map = {eid: idx for idx, eid in enumerate(self.mesh_element_ids)}

        total_elems = self.mesh_conn_np.shape[0]
        elem_stress = np.zeros((total_elems, 6), dtype=float)
        valid_mask = np.zeros(total_elems, dtype=bool)

        for res in elem_results:
            if res is None:
                continue
            if hasattr(res, "id"):
                eid = getattr(res, "id", None)
            elif isinstance(res, dict):
                eid = res.get("id")
            else:
                eid = None
            if hasattr(res, "stress"):
                stress_src = getattr(res, "stress", None)
            elif isinstance(res, dict):
                stress_src = res.get("stress")
            else:
                stress_src = None
            if eid is None or stress_src is None:
                continue
            elem_idx = self.element_index_map.get(eid)
            if elem_idx is None or elem_idx >= total_elems:
                continue
            stress_arr = self._sanitize_array(stress_src).flatten()
            nvals = min(6, stress_arr.size)
            if nvals == 0:
                continue
            elem_stress[elem_idx, :nvals] = stress_arr[:nvals]
            valid_mask[elem_idx] = True

        if not np.any(valid_mask):
            return False

        centers = self.mesh_coords_np[self.mesh_conn_np[valid_mask]].mean(axis=1)
        sigma_xx = elem_stress[valid_mask, 0]
        sigma_vm = self._von_mises_array(elem_stress[valid_mask])

        disps = np.asarray(self.last_displacements, dtype=float)
        if disps.ndim != 2 or disps.shape[1] != 3:
            disps = disps.reshape(-1, 3)
        if disps.shape[0] != self.mesh_coords_np.shape[0]:
            return False
        disp_mag = np.linalg.norm(disps, axis=1)

        from matplotlib import pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))

        sc0 = axs[0].scatter(centers[:, 0], centers[:, 1], c=sigma_xx, cmap="viridis", s=120, edgecolors="k")
        axs[0].set_title("sigma_xx [MPa]")
        axs[0].set_xlabel("x [mm]")
        axs[0].set_ylabel("y [mm]")
        fig.colorbar(sc0, ax=axs[0], fraction=0.046)

        sc1 = axs[1].scatter(centers[:, 0], centers[:, 1], c=sigma_vm, cmap="plasma", s=120, edgecolors="k")
        axs[1].set_title("von Mises [MPa]")
        axs[1].set_xlabel("x [mm]")
        axs[1].set_ylabel("y [mm]")
        fig.colorbar(sc1, ax=axs[1], fraction=0.046)

        sc2 = axs[2].scatter(self.mesh_coords_np[:, 0], self.mesh_coords_np[:, 1], c=disp_mag, cmap="coolwarm", s=100, edgecolors="k")
        axs[2].set_title("|u| [mm]")
        axs[2].set_xlabel("x [mm]")
        axs[2].set_ylabel("y [mm]")
        fig.colorbar(sc2, ax=axs[2], fraction=0.046)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=200)
        plt.close(fig)
        return True

    def export_frame_snapshot(self, step: int, current_time: float) -> None:
        """Export a 2D scatter snapshot for the active field at a given time step."""
        if not self.export_frames or not HAS_MATPLOTLIB:
            return
        if self.mesh_coords_np is None or self.last_displacements is None:
            return
        scalars, title, cmap = self._resolve_field_scalars(self.display_field)
        if scalars is None:
            return
        coords = np.asarray(self.mesh_coords_np, dtype=float)
        if coords.shape[0] != scalars.shape[0]:
            return

        field_label = title or self.display_field
        slug = self._field_slug(field_label)
        export_dir = Path(_root_dir) / "exports" / "frames" / slug
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = export_dir / f"{slug}_step{step:05d}_t{current_time:.5f}s.png"

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 2.5))
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scalars,
            cmap=cmap,
            s=18,
            edgecolors="none",
        )
        ax.set_title(f"{field_label}  t = {current_time:.4f} s")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", "box")
        fig.colorbar(sc, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(str(filename), dpi=220)
        plt.close(fig)
        self.status_bar.showMessage(f"Frame exported: {filename.name}")

    def _von_mises(self, stress_vec):
        sxx, syy, szz, sxy, sxz, syz = self._sanitize_array(stress_vec)
        return math.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )

    def _von_mises_array(self, stress):
        stress = self._sanitize_array(stress)
        sxx = stress[:, 0]
        syy = stress[:, 1]
        szz = stress[:, 2]
        sxy = stress[:, 3]
        sxz = stress[:, 4]
        syz = stress[:, 5]
        return np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )

    def _sanitize_array(self, arr, limit=1e12):
        arr = np.asarray(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
        np.clip(arr, -limit, limit, out=arr)
        return arr

    def _field_slug(self, text: str) -> str:
        slug = re.sub(r'[^0-9a-zA-Z]+', '-', text.lower()).strip('-')
        return slug or "field"

    def _export_pyvista_snapshot(self, output_path: Path) -> bool:
        """Render a deformed stress view off-screen via PyVista."""
        if not HAS_PYVISTA:
            return False
        if self.mesh_coords_np is None or self.mesh_conn_np is None:
            return False
        if self.last_displacements is None:
            return False
        try:
            coords = np.asarray(self.mesh_coords_np, dtype=float)
            conn = np.asarray(self.mesh_conn_np, dtype=np.int64)
            disps = np.asarray(self.last_displacements, dtype=float)
        except Exception:
            return False
        if coords.ndim != 2 or coords.shape[1] != 3:
            return False
        if disps.shape != coords.shape:
            try:
                disps = disps.reshape(coords.shape)
            except Exception:
                return False
        n_per_elem = conn.shape[1] if conn.ndim == 2 else 0
        if n_per_elem not in (4, 8):
            return False
        import pyvista as pv
        try:
            cell_type = pv.CellType.TETRA if n_per_elem == 4 else pv.CellType.HEXAHEDRON
        except AttributeError:
            return False
        cells = np.hstack([
            np.full((conn.shape[0], 1), n_per_elem, dtype=np.int64),
            conn.astype(np.int64)
        ]).ravel()
        cell_types = np.full(conn.shape[0], cell_type, dtype=np.uint8)
        scale = getattr(self, "deformation_scale", 1.0)
        disps = self._sanitize_array(disps, limit=1e6)
        points = coords + disps * float(scale)
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        scalars = self.last_vm
        if scalars is None:
            scalars = self.compute_nodal_von_mises()
        if scalars is not None:
            scalars = self._sanitize_array(scalars)
            grid.point_data["Von Mises"] = scalars

        plotter = pv.Plotter(off_screen=True, window_size=(1280, 720))
        kwargs = {"show_edges": True}
        if scalars is not None:
            kwargs.update({
                "scalars": scalars,
                "cmap": "inferno",
                "scalar_bar_args": {"title": "Von Mises Stress (Pa)"}
            })
        else:
            kwargs.update({"color": "lightblue"})
        plotter.add_mesh(grid, **kwargs)
        plotter.view_xy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plotter.show(auto_close=True, screenshot=str(output_path))
        finally:
            plotter.close()
        return True

    def compute_nodal_von_mises(self, components=None):
        if self.solver is None:
            return None
        comps = components if components is not None else self.last_stress_components
        if comps is None:
            comps = self.compute_nodal_stress_components()
            if comps is None:
                return None
            self.last_stress_components = comps
        return self._von_mises_array(comps)

    def compute_nodal_stress_components(self):
        if self.solver is None or self.mesh_conn_np is None or self.mesh_coords_np is None:
            return None
        try:
            elem_results = self.solver.get_element_results()
        except Exception:
            return None
        if not elem_results:
            return None

        num_nodes = self.mesh_coords_np.shape[0]
        if num_nodes == 0:
            return None

        if not self.element_index_map and self.mesh_element_ids is not None:
            self.element_index_map = {eid: idx for idx, eid in enumerate(self.mesh_element_ids)}

        totals = np.zeros((num_nodes, 6), dtype=float)
        counts = np.zeros(num_nodes, dtype=float)

        for res in elem_results:
            if res is None:
                continue
            if hasattr(res, "id"):
                eid = getattr(res, "id", None)
            elif isinstance(res, dict):
                eid = res.get("id")
            else:
                eid = None
            if hasattr(res, "stress"):
                stress_src = getattr(res, "stress", None)
            elif isinstance(res, dict):
                stress_src = res.get("stress")
            else:
                stress_src = None
            if eid is None or stress_src is None:
                continue
            elem_idx = self.element_index_map.get(eid)
            if elem_idx is None or elem_idx >= len(self.mesh_conn_np):
                continue
            stress_vec = np.zeros(6, dtype=float)
            stress_arr = self._sanitize_array(stress_src).flatten()
            nvals = min(6, stress_arr.size)
            if nvals > 0:
                stress_vec[:nvals] = stress_arr[:nvals]
            node_indices = self.mesh_conn_np[elem_idx]
            totals[node_indices] += stress_vec
            counts[node_indices] += 1.0

        mask = counts == 0.0
        counts[mask] = 1.0
        totals[mask] = 0.0
        return totals / counts[:, None]

    def update_visualization(self, displacements, stress_vm):
        if not HAS_PYVISTA or not hasattr(self.viewer, "update_field"):
            return
        scalars = stress_vm if stress_vm is not None else None
        if scalars is not None:
            scalars = self._sanitize_array(scalars)
        self.viewer.update_field(
            displacements=displacements,
            scalars=scalars,
            title="Von Mises Stress (Pa)",
            scale=self.deformation_scale,
            cmap="inferno",
            show_scalar_bar=self.show_scalar_bar
        )

    def update_visualization_from_cache(self):
        if not self.enable_live_view:
            return
        if self.last_displacements is None or len(self.last_displacements) == 0:
            return
        scalars, title, cmap = self._resolve_field_scalars(self.display_field)
        show_bar = self.show_scalar_bar and scalars is not None
        self.viewer.update_field(
            displacements=self.last_displacements,
            scalars=scalars,
            title=title,
            scale=self.deformation_scale,
            cmap=cmap,
            show_scalar_bar=show_bar
        )

    def _resolve_field_scalars(self, field_name):
        """Return (scalars, title, cmap) for the requested display field."""
        scalars = None
        title = ""
        cmap = "inferno"

        if field_name == "Von Mises Stress":
            scalars = self.last_vm if self.last_vm is not None else self.compute_nodal_von_mises()
            if scalars is not None:
                self.last_vm = scalars
                title = "Von Mises Stress (Pa)"
        elif field_name == "Displacement Magnitude":
            disp = np.asarray(self.last_displacements, dtype=float)
            if disp.ndim == 2 and disp.shape[0] > 0:
                scalars = np.linalg.norm(disp[:, :3], axis=1)
                title = "Displacement Magnitude (m)"
                cmap = "viridis"
        else:
            comps = self.compute_nodal_stress_components()
            if comps is not None:
                mapping = {
                    "Sxx": 0,
                    "Syy": 1,
                    "Szz": 2,
                    "Sxy": 3,
                    "Sxz": 4,
                    "Syz": 5,
                }
                idx = mapping.get(field_name)
                if idx is not None:
                    scalars = comps[:, idx]
                    title = f"{field_name} (Pa)"
                    cmap = "plasma"

        if scalars is not None:
            scalars = self._sanitize_array(scalars)
        return scalars, title, cmap

    def on_field_changed(self, text):
        self.display_field = text
        self.update_visualization_from_cache()

    def on_scale_changed(self):
        """Update deformation scale factor from toolbar control."""
        if hasattr(self, "scale_spin"):
            self.deformation_scale = self.scale_spin.value()
            self.update_visualization_from_cache()

    def on_gpu_toggled(self, checked):
        """Enable or disable GPU solver usage."""
        if not HAS_GPU_SOLVER:
            if hasattr(self, "gpu_checkbox"):
                self.gpu_checkbox.setChecked(False)
            QMessageBox.warning(self, "GPU Solver Unavailable",
                                "Rebuild FEM-ML with CUDA support (FEMML_USE_GPU=ON) to enable GPU runs.")
            self.use_gpu = False
            return
        self.use_gpu = bool(checked)

    def on_legend_toggled(self, checked):
        """Toggle scalar-bar visibility."""
        self.show_scalar_bar = bool(checked)
        self.update_visualization_from_cache()

    def on_live_view_toggled(self, checked):
        """Enable/disable PyVista live updates."""
        self.enable_live_view = bool(checked)
        if self.enable_live_view:
            self.update_visualization_from_cache()

    def on_frame_export_toggled(self, checked):
        """Enable/disable time-step frame exports."""
        if checked and not HAS_MATPLOTLIB:
            QMessageBox.warning(
                self,
                "Matplotlib Required",
                "Install matplotlib to enable frame exports (pip install matplotlib).",
            )
            self.frame_export_checkbox.setChecked(False)
            return
        self.export_frames = bool(checked)
        if hasattr(self, "frame_interval_spin"):
            self.frame_interval_spin.setEnabled(self.export_frames)
        if self.export_frames:
            export_dir = Path(_root_dir) / "exports" / "frames"
            export_dir.mkdir(parents=True, exist_ok=True)
            self.status_bar.showMessage(
                f"Frame export enabled (every {self.frame_export_interval} steps)"
            )
        else:
            self.status_bar.showMessage("Frame export disabled")

    def on_frame_interval_changed(self, value):
        """Update frame export interval from toolbar control."""
        self.frame_export_interval = max(1, int(value))
        if self.export_frames:
            self.status_bar.showMessage(
                f"Frame export interval set to {self.frame_export_interval} steps"
            )

    def show_howto(self):
        """Show how-to guide"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("How to Run")
        msg.setText("FEM-ML Quick Guide")
        msg.setInformativeText(textwrap.dedent("""\
            Step 1: Import Mesh
              File -> Import Mesh
              Select: examples/simple_cube/cube.inp

            Step 2: Create Material
              Model -> Create Material
              Select preset: Aluminum

            Step 3: Create Boundary Condition
              Model -> Create Boundary Condition
              Nodes: 1,2,5,6 (bottom face)
              Type: Fixed

            Step 4: Create Load
              Model -> Create Load
              Nodes: 3,4,7,8
              Direction: Y, Value: 5e5 N

            Step 5: Run
              Job -> Submit Job (or press F5)
              Watch real-time results!

            For more help, see HOW_TO_RUN.md
        """))
        msg.setDetailedText(textwrap.dedent("""\
            Full Documentation:
            - HOW_TO_RUN.md - Complete step-by-step guide
            - QUICKSTART.md - 15-minute tutorial
            - README.md - Full reference
            - GET_STARTED.md - Quick reference

            Example Location:
            D:/Research-work/Examples/FEM-ML/examples/simple_cube
        """))
        msg.exec_()

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About FEM-ML",
            "<h2>FEM-ML</h2>"
            "<p><b>Explicit FEM Solver v1.0</b></p>"
            "<p>A high-performance explicit finite element solver with:</p>"
            "<ul>"
            "<li>Multi-element meshing</li>"
            "<li>Neural network integration</li>"
            "<li>Python & C++ APIs</li>"
            "<li>Abaqus-like interface</li>"
            "</ul>"
            "<p>Built with C++17, Python, Qt, and PyVista</p>"
            "<p>© 2025 FEM-ML Project</p>"
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FEM-ML")
    app.setOrganizationName("FEM-ML")

    # Set style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
