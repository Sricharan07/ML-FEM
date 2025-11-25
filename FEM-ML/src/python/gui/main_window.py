#!/usr/bin/env python3
"""
FEM-ML GUI: Enhanced Main Application Window
Abaqus CAE-like interface for explicit FEM simulations
"""

import sys
import os
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
_root_dir = os.path.abspath(os.path.join(_script_dir, "../../../"))
_candidates = [
    os.path.join(_root_dir, "build"),
    os.path.join(_root_dir, "build", "src", "python"),
    os.path.join(_root_dir, "build", "src", "python", "Release"),
    os.path.join(_root_dir, "build", "src", "python", "Debug"),
    os.path.join(_root_dir, "build", "Release"),
    os.path.join(_root_dir, "build", "Debug"),
]
for path in _candidates:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    import femml
    HAS_FEMML = True
except ImportError:
    HAS_FEMML = False
    print("="*60)
    print("WARNING: femml module not found!")
    print("Please build the project first:")
    print("  cd FEM-ML")
    print("  .\\build.ps1")
    print("="*60)


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

        # BC type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Fixed (U1=U2=U3=0)", "Fixed X", "Fixed Y", "Fixed Z"])
        layout.addRow("Type:", self.type_combo)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.setLayout(layout)

    def get_bc(self):
        """Return BC parameters"""
        nodes_text = self.nodes_edit.text()
        nodes = parse_node_list(nodes_text)

        component_map = {
            0: -1,  # All
            1: 0,   # X
            2: 1,   # Y
            3: 2,   # Z
        }

        return {
            'name': self.name_edit.text(),
            'nodes': nodes,
            'component': component_map[self.type_combo.currentIndex()]
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
        for res in element_results:
            stress = list(res.stress) if hasattr(res, "stress") else res["stress"]
            strain = list(res.strain) if hasattr(res, "strain") else res["strain"]
            rows.append([
                res.id if hasattr(res, "id") else res["id"],
                res.type if hasattr(res, "type") else res["type"],
                res.volume if hasattr(res, "volume") else res["volume"],
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
        else:
            # Fallback UI
            layout = QVBoxLayout()
            label = QLabel("3D Visualization\n(PyVista not available)")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self.setLayout(layout)

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

        self.clear()
        self.mesh_actor = self.add_mesh(
            self.mesh_data,
            show_edges=True,
            color='lightblue',
            opacity=0.9,
            edge_color='black'
        )

        self.reset_camera()
        self.update()

    def update_field(self, displacements=None, scalars=None, title="", scale=1.0, cmap="viridis"):
        """Update visualization with optional displacement field and scalars."""
        if not HAS_PYVISTA or self.mesh_data is None:
            return

        frame = self.mesh_data.copy()
        if self.base_points is not None:
            frame.points = self.base_points.copy()
        if displacements is not None and self.base_points is not None:
            disp_array = np.array(displacements) * scale
            if disp_array.shape == self.base_points.shape:
                frame.points = self.base_points + disp_array

        plot_kwargs = {
            'show_edges': True,
        }

        if scalars is not None:
            scalars = np.asarray(scalars)
            if scalars.shape[0] == frame.n_points:
                plot_kwargs.update({
                    'scalars': scalars,
                    'cmap': cmap,
                    'show_scalar_bar': True,
                    'scalar_bar_args': {'title': title or '', 'vertical': True},
                })
            else:
                frame.clear_data()
                frame.cell_data["field"] = scalars
                plot_kwargs.update({
                    'scalars': scalars,
                    'cmap': cmap,
                    'show_scalar_bar': True,
                    'scalar_bar_args': {'title': title or '', 'vertical': True},
                })
        else:
            plot_kwargs.update({
                'color': 'lightblue',
                'opacity': 0.9,
                'edge_color': 'black',
                'show_scalar_bar': False,
            })

        self.clear()
        self.mesh_actor = self.add_mesh(frame, **plot_kwargs)
        if title:
            self.add_text(title, font_size=10)
        self.update()

    def update_displacements(self, displacements, scale=1000.0):
        disp_mag = np.linalg.norm([[d[0], d[1], d[2]] for d in displacements], axis=1)
        self.update_field(displacements=displacements, scalars=disp_mag,
                          title="Displacement (m)", scale=scale, cmap="jet")


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
        self.is_solving = False
        self.solver_params = None
        self.mesh_nodes_cache = None
        self.display_field = "Von Mises Stress"
        self.last_displacements = None
        self.last_stress_components = None
        self.last_vm = None

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
            QTreeWidgetItem(self.model_tree.bcs_root,
                [f"{bc_data['name']} ({len(bc_data['nodes'])} nodes)"])
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
            self.status_bar.showMessage("Starting analysis...")

            # Create solver
            self.solver = femml.ExplicitSolver(self.mesh)

            # Assign material (use first one)
            mat_name = list(self.materials.keys())[0]
            self.solver.set_material(self.materials[mat_name])

            # Add BCs
            for bc_data in self.bcs:
                bc = femml.BoundaryCondition()
                bc.type = femml.BCType.FIXED
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
            self.last_stress_components = self.compute_nodal_stress_components()
            self.update_visualization_from_cache()

            self.update_results_views()
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

        # Update visualization every 10 steps
        if step % 10 == 0:
            disps = self.solver.get_displacements()
            self.last_displacements = disps
            self.last_stress_components = self.compute_nodal_stress_components()
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

    def _von_mises(self, stress_vec):
        sxx, syy, szz, sxy, sxz, syz = stress_vec
        return math.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )

    def _von_mises_array(self, stress):
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

    def compute_nodal_von_mises(self):
        if self.mesh is None or self.solver is None:
            return None
        comps = self.last_stress_components
        if comps is None:
            comps = self.compute_nodal_stress_components()
        if comps is None:
            return None
        return self._von_mises_array(comps)

    def compute_nodal_stress_components(self):
        if self.mesh is None or self.solver is None:
            return None
        if self.last_stress_components is not None:
            return self.last_stress_components

        try:
            elem_results = self.solver.get_element_results()
        except Exception:
            return None

        num_nodes = self.mesh.get_num_nodes()
        sums = np.zeros((num_nodes, 6))
        counts = np.zeros(num_nodes)

        for res in elem_results:
            stress = np.array(res.stress)
            try:
                elem = self.mesh.get_element(res.id)
            except Exception:
                continue
            for node_id in getattr(elem, "nodes", []):
                idx = node_id - 1
                if 0 <= idx < num_nodes:
                    sums[idx] += stress
                    counts[idx] += 1

        mask = counts == 0
        counts[mask] = 1
        sums[mask] = 0.0
        self.last_stress_components = sums / counts[:, None]
        return self.last_stress_components

    def update_visualization(self, displacements, stress_vm):
        if not HAS_PYVISTA or not hasattr(self.viewer, "update_field"):
            return
        scalars = stress_vm if stress_vm is not None else None
        self.viewer.update_field(
            displacements=displacements,
            scalars=scalars,
            title="Von Mises Stress (Pa)",
            scale=1000.0,
            cmap="inferno"
        )

    def update_visualization_from_cache(self):
        if self.last_displacements is None:
            return
        field = self.display_field
        title = ""
        cmap = "inferno"
        scalars = None

        if field == "Von Mises Stress":
            scalars = self.compute_nodal_von_mises()
            title = "Von Mises Stress (Pa)"
        elif field == "Displacement Magnitude":
            disp = np.array([[d[0], d[1], d[2]] for d in self.last_displacements])
            scalars = np.linalg.norm(disp, axis=1)
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
                idx = mapping.get(field)
                if idx is not None:
                    scalars = comps[:, idx]
                    title = f"{field} (Pa)"
                    cmap = "plasma"

        if scalars is not None:
            self.viewer.update_field(
                displacements=self.last_displacements,
                scalars=scalars,
                title=title,
                scale=1000.0,
                cmap=cmap
            )

    def on_field_changed(self, text):
        self.display_field = text
        self.update_visualization_from_cache()

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
