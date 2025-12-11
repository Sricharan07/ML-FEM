from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .mesh import AbaqusInpImporter as CoreImporter
from .mesh import Mesh as CoreMesh
from logic import (
    BoundaryCondition as LogicBoundaryCondition,
    MaterialProperties,
    PyTorchExplicitSolver,
    SimulationConfig,
)


# ---------------------------------------------------------------------------
# Mesh adapter matching the legacy FEM-ML API
# ---------------------------------------------------------------------------
@dataclass
class _NodeAdapter:
    id: int
    coords: Sequence[float]


@dataclass
class _ElementAdapter:
    id: int
    nodes: List[int]
    type: str = "C3D8"


class MeshAdapter:
    """Adapter that mimics the C++ femml mesh API for the GUI."""

    def __init__(self, mesh: CoreMesh) -> None:
        self._mesh = mesh
        self._node_ids = mesh.node_ids()
        self._element_ids = mesh.element_ids()
        self._node_dict = mesh.nodes
        self._element_dict = mesh.elements

    def get_num_nodes(self) -> int:
        return len(self._node_ids)

    def get_num_elements(self) -> int:
        return len(self._element_ids)

    def get_node(self, idx: int) -> _NodeAdapter:
        """Access node by actual id or sequential index (1-based)."""
        if idx in self._node_dict:
            node = self._node_dict[idx]
            return _NodeAdapter(id=idx, coords=node.coords)
        seq = idx - 1
        if 0 <= seq < len(self._node_ids):
            node_id = self._node_ids[seq]
            node = self._node_dict[node_id]
            return _NodeAdapter(id=node_id, coords=node.coords)
        raise IndexError(f"Node {idx} not found")

    def get_element(self, idx: int) -> _ElementAdapter:
        if idx in self._element_dict:
            elem = self._element_dict[idx]
            return _ElementAdapter(id=idx, nodes=list(elem.nodes), type=elem.type)
        seq = idx - 1
        if 0 <= seq < len(self._element_ids):
            eid = self._element_ids[seq]
            elem = self._element_dict[eid]
            return _ElementAdapter(id=eid, nodes=list(elem.nodes), type=elem.type)
        raise IndexError(f"Element {idx} not found")

    def node_ids(self) -> List[int]:
        return list(self._node_ids)

    def element_ids(self) -> List[int]:
        return list(self._element_ids)

    def to_numpy(self):
        coords, conn, node_ids, element_ids = self._mesh.to_numpy()
        return coords, conn, node_ids, element_ids

    def summary(self):
        return self._mesh.summary()


class AbaqusImporter:
    def __init__(self) -> None:
        self._importer = CoreImporter(verbose=False)

    def import_mesh(self, filename: str) -> MeshAdapter:
        mesh = self._importer.import_file(filename)
        return MeshAdapter(mesh)


# ---------------------------------------------------------------------------
# Legacy class shims
# ---------------------------------------------------------------------------
@dataclass
class LinearElastic:
    name: str
    E: float  # Pa
    nu: float
    rho: float  # kg/m^3 (converted later)


@dataclass
class NeuralMaterial:
    name: str
    rho: float


class BCType(Enum):
    DISPLACEMENT = auto()
    FIXED = auto()


class LoadType(Enum):
    FORCE = auto()


@dataclass
class BoundaryCondition:
    nodes: List[int] = field(default_factory=list)
    component: int = -1
    type: BCType = BCType.FIXED
    value: float = 0.0
    ramp_time: float = 0.0


@dataclass
class Load:
    nodes: List[int] = field(default_factory=list)
    component: int = 0
    value: float = 0.0
    type: LoadType = LoadType.FORCE


@dataclass
class SolverParams:
    time_step: float = 5e-6
    num_steps: int = 1000
    damping: float = 0.0
    output_interval: int = 50
    auto_time_step: bool = True


@dataclass
class NodeResult:
    id: int
    coords: Sequence[float]
    displacement: Sequence[float]


@dataclass
class ElementResult:
    id: int
    stress: Sequence[float]


# ---------------------------------------------------------------------------
# Explicit solver wrapper
# ---------------------------------------------------------------------------
class ExplicitSolver:
    def __init__(self, mesh: MeshAdapter) -> None:
        self.mesh = mesh
        self.material: Optional[LinearElastic] = None
        self.boundary_conditions: List[BoundaryCondition] = []
        self.loads: List[Load] = []
        self.params = SolverParams()
        self._result: Optional[SimulationResults] = None
        self._displacements = None
        self._node_results: List[NodeResult] = []
        self._element_results: List[ElementResult] = []
        self._finished = False
        self._current_step = 0
        self._current_time = 0.0
        self._node_id_to_index: Dict[int, int] = {}
        self._node_ids: List[int] = []
        self._element_ids: List[int] = []
        self._thread: Optional[threading.Thread] = None
        self._thread_done = False
        self._pending_finalize = False
        self._worker_error: Optional[Exception] = None
        self._progress_lock = threading.Lock()
        self._progress_info = {"step": 0, "time": 0.0, "dt": 0.0, "ke": 0.0, "ie": 0.0}
        self._live_displacements: Optional[np.ndarray] = None
        self._live_element_results: List[ElementResult] = []

    def set_material(self, material: LinearElastic) -> None:
        self.material = material

    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        self.boundary_conditions.append(bc)

    def add_load(self, load: Load) -> None:
        # Loads are currently ignored by the PyTorch solver but stored for completeness.
        self.loads.append(load)

    def set_parameters(self, params: SolverParams) -> None:
        self.params = params

    def initialize(self) -> None:
        if self.material is None:
            raise RuntimeError("Material must be set before initializing the solver")

        coords_np, conn_np, node_ids, element_ids = self.mesh.to_numpy()
        self._node_ids = node_ids
        self._element_ids = element_ids
        self._node_id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
        self._live_displacements = None
        self._live_element_results = []

        coords_t = torch.tensor(coords_np, dtype=torch.float64)
        elements_t = torch.tensor(conn_np, dtype=torch.long)

        total_time = max(1, self.params.num_steps) * max(self.params.time_step, 1e-8)
        config = SimulationConfig(
            total_time=total_time,
            target_time_step=self.params.time_step if self.params.auto_time_step else self.params.time_step,
            safety_factor=0.5,
            max_mass_scale=10.0,
            print_mass_scaling=False,
        )
        density = max(self.material.rho * 1e-12, 1e-12)  # kg/m^3 -> tonne/mm^3
        youngs_modulus = self.material.E / 1e6  # Pa -> MPa
        material = MaterialProperties(
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=self.material.nu,
        )

        solver = PyTorchExplicitSolver(coords_t, elements_t, material=material, config=config)
        logic_bcs: List[LogicBoundaryCondition] = []
        for idx, bc in enumerate(self.boundary_conditions):
            node_indices = [
                self._node_id_to_index[nid]
                for nid in bc.nodes
                if nid in self._node_id_to_index
            ]
            components = (bc.component,) if bc.component >= 0 else (-1,)
            value_mm = bc.value * 1e3 if bc.type == BCType.DISPLACEMENT else 0.0
            logic_bcs.append(
                LogicBoundaryCondition(
                    name=f"BC-{idx+1}",
                    node_indices=node_indices,
                    components=components,
                    bc_type="displacement" if bc.type == BCType.DISPLACEMENT else "fixed",
                    value=value_mm,
                    ramp_time=bc.ramp_time,
                )
            )

        self._finished = False
        self._thread_done = False
        self._pending_finalize = False
        self._worker_error = None

        def progress_callback(info):
            with self._progress_lock:
                self._progress_info = {
                    "step": int(info.get("step", self._progress_info["step"])),
                    "time": float(info.get("time", self._progress_info["time"])),
                    "dt": float(info.get("dt", self._progress_info["dt"])),
                    "ke": float(info.get("ke", self._progress_info["ke"])),
                    "ie": float(info.get("ie", self._progress_info["ie"])),
                }
                disp = info.get("displacements")
                if disp is not None:
                    if isinstance(disp, torch.Tensor):
                        disp = disp.detach().cpu().numpy()
                    self._live_displacements = np.array(disp, dtype=float)
                elem_stress = info.get("element_stress")
                if elem_stress is not None:
                    if isinstance(elem_stress, torch.Tensor):
                        elem_stress = elem_stress.detach().cpu().numpy()
                    elem_arr = np.array(elem_stress, dtype=float)
                    self._live_element_results = [
                        ElementResult(id=eid, stress=elem_arr[idx].tolist())
                        for idx, eid in enumerate(self._element_ids[: elem_arr.shape[0]])
                    ]

        def worker():
            try:
                result = solver.run(logic_bcs, progress_callback=progress_callback)
                self._result = result
                self._displacements = (
                    result.displacements.view(len(node_ids), 3).cpu().numpy()
                )
                element_stress = result.element_stress.cpu().numpy()
                self._node_results = [
                    NodeResult(
                        id=nid,
                        coords=self.mesh.get_node(nid).coords,
                        displacement=self._displacements[idx],
                    )
                    for idx, nid in enumerate(node_ids)
                ]
                self._element_results = [
                    ElementResult(id=eid, stress=element_stress[idx].tolist())
                    for idx, eid in enumerate(element_ids)
                ]
                self._current_step = len(result.time_history)
                self._current_time = (
                    result.time_history[-1] if result.time_history else 0.0
                )
                with self._progress_lock:
                    self._progress_info.update(
                        {"step": self._current_step, "time": self._current_time}
                    )
                self._pending_finalize = True
            except Exception as exc:
                self._worker_error = exc
            finally:
                self._thread_done = True

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    # --- FEM-ML solver API ---------------------------------------------
    def step(self) -> None:
        if self._worker_error:
            err = self._worker_error
            self._worker_error = None
            raise RuntimeError(str(err))
        if self._pending_finalize:
            self._pending_finalize = False
            self._finished = True
            return
        if not self._thread_done:
            return

    def is_finished(self) -> bool:
        return self._finished

    def get_current_step(self) -> int:
        if self._finished:
            return self._current_step
        with self._progress_lock:
            return int(self._progress_info.get("step", 0))

    def get_current_time(self) -> float:
        if self._finished:
            return self._current_time
        with self._progress_lock:
            return float(self._progress_info.get("time", 0.0))

    def get_displacements(self) -> List[List[float]]:
        with self._progress_lock:
            if self._displacements is not None:
                return self._displacements.tolist()
            if self._live_displacements is not None:
                return self._live_displacements.tolist()
            return [[0.0, 0.0, 0.0] for _ in self._node_ids]

    def get_displacement_magnitude(self) -> np.ndarray:
        disps = self.get_displacements()
        if not disps:
            return np.array([])
        disp_arr = np.array(disps)
        if disp_arr.ndim != 2 or disp_arr.shape[1] != 3:
            return np.array([])
        return np.linalg.norm(disp_arr, axis=1)

    def get_kinetic_energy(self) -> float:
        if not self._result or not self._result.kinetic_energy:
            with self._progress_lock:
                return float(self._progress_info.get("ke", 0.0))
        return float(self._result.kinetic_energy[-1])

    def get_strain_energy(self) -> float:
        if not self._result or not self._result.internal_energy:
            with self._progress_lock:
                return float(self._progress_info.get("ie", 0.0))
        return float(self._result.internal_energy[-1])

    def get_total_energy(self) -> float:
        return self.get_kinetic_energy() + self.get_strain_energy()

    def get_element_results(self) -> List[ElementResult]:
        if self._element_results and self._finished:
            return list(self._element_results)
        if self._live_element_results:
            return list(self._live_element_results)
        return []

    def get_node_results(self) -> List[NodeResult]:
        return list(self._node_results)

    def write_results(self, filename: str) -> None:
        if self._displacements is None:
            return
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["node_id", "ux", "uy", "uz"])
            for node_result in self._node_results:
                ux, uy, uz = node_result.displacement
                writer.writerow([node_result.id, ux, uy, uz])


# GPU solver shim
GPUExplicitSolver = ExplicitSolver


__all__ = [
    "AbaqusImporter",
    "BCType",
    "BoundaryCondition",
    "ElementResult",
    "ExplicitSolver",
    "GPUExplicitSolver",
    "LinearElastic",
    "Load",
    "LoadType",
    "MeshAdapter",
    "NeuralMaterial",
    "NodeResult",
    "SolverParams",
]
