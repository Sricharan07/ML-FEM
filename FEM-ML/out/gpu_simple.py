#!/usr/bin/env python3
"""Run the prescribed-displacement cube test with optional GPU acceleration."""

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_CANDIDATES = [
    ROOT / "build",
    ROOT / "build" / "src" / "python",
    ROOT / "build" / "src" / "python" / "Release",
    ROOT / "build" / "src" / "python" / "Debug",
]
for candidate in PYTHON_CANDIDATES:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import femml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Unable to import femml module. Build the project first (see README).\n"
        f"Details: {exc}"
    ) from exc


def build_solver(mesh):
    """Return GPU solver if available, otherwise fallback to CPU."""
    if hasattr(femml, "GPUExplicitSolver"):
        solver = femml.GPUExplicitSolver(mesh)
        try:
            solver.set_device_id(0)
        except AttributeError:
            pass
        print("   Using GPUExplicitSolver")
        return solver
    print("   Using CPU ExplicitSolver")
    return femml.ExplicitSolver(mesh)


def main() -> None:
    mesh_path = ROOT / "examples" / "simple_cube" / "cube.inp"
    print("=" * 72)
    print(" Prescribed-Displacement Cube (2x2x1 elements, 1 mm^3)")
    print("=" * 72)

    importer = femml.AbaqusImporter()
    mesh = importer.import_mesh(str(mesh_path))
    print(f"[mesh] Nodes: {mesh.get_num_nodes()}, Elements: {mesh.get_num_elements()}")

    solver = build_solver(mesh)
    solver.set_material(femml.create_aluminum())

    left_nodes = [1, 4, 7, 10, 13, 16]
    right_nodes = [3, 6, 9, 12, 15, 18]
    target_disp = 5.43e-6
    ramp_time = 0.002

    left_bc = femml.BoundaryCondition()
    left_bc.type = femml.BCType.FIXED
    left_bc.nodes = left_nodes
    left_bc.component = -1
    solver.add_boundary_condition(left_bc)

    right_bc = femml.BoundaryCondition()
    right_bc.type = femml.BCType.DISPLACEMENT
    right_bc.nodes = right_nodes
    right_bc.component = 0
    right_bc.value = target_disp
    right_bc.ramp_time = ramp_time
    solver.add_boundary_condition(right_bc)

    params = femml.SolverParams()
    params.time_step = 5e-7
    params.num_steps = int(math.ceil(ramp_time / params.time_step))
    params.output_interval = max(1, params.num_steps // 10)
    params.auto_time_step = False
    params.damping = 0.0
    solver.set_parameters(params)

    print("[solver] Initializing ...")
    solver.initialize()

    history_time = []
    history_disp = []

    def cb(step: int, t: float) -> None:
        if step == 0:
            return
        avg_disp = np.mean([solver.get_node_displacement(n)[0] for n in right_nodes])
        history_time.append(t)
        history_disp.append(avg_disp)

    solver.set_output_callback(cb)

    print("[solver] Solving ...")
    solver.solve()
    print("[solver] Done.")

    out_dir = Path(__file__).resolve().parent
    results_path = out_dir / "gpu_simple_results.csv"
    solver.write_results(str(results_path))
    print(f"[results] Saved nodal CSV -> {results_path}")

    if history_time:
        final_disp = history_disp[-1] * 1e3
        print(
            f"[summary] Final average X displacement on right face: {final_disp:.5f} mm"
        )


if __name__ == "__main__":
    main()
