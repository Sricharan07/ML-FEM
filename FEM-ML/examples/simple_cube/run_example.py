#!/usr/bin/env python3
"""Simple cube tension test with prescribed displacement boundary conditions."""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../../"))
_PYTHON_CANDIDATES = [
    os.path.join(_ROOT, "build"),
    os.path.join(_ROOT, "build", "src", "python"),
    os.path.join(_ROOT, "build", "src", "python", "Release"),
    os.path.join(_ROOT, "build", "src", "python", "Debug"),
]
for _path in _PYTHON_CANDIDATES:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import femml  # type: ignore
except ImportError:
    print("Error: femml module not found. Please build the project first.")
    sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("FEM-ML Simple Cube Example")
    print("=" * 60)

    print("\n1. Importing mesh...")
    importer = femml.AbaqusImporter()
    cube_path = os.path.join(_HERE, "cube.inp")
    mesh = importer.import_mesh(cube_path)

    print("\n2. Creating material...")
    material = femml.create_aluminum()
    print(f"   Material: {material.get_name()}")
    print(f"   Density: {material.get_density()} kg/m^3")

    print("\n3. Setting up solver...")
    solver = femml.ExplicitSolver(mesh)
    solver.set_material(material)

    left_nodes = [1, 4, 7, 10, 13, 16]   # x = 0 mm plane
    right_nodes = [3, 6, 9, 12, 15, 18]  # x = 1 mm plane
    target_disp = 5.43e-6  # 0.00543 mm
    ramp_time = 0.002      # seconds
    disp_rate = target_disp / ramp_time

    # Clamp the left end in all directions
    left_bc = femml.BoundaryCondition()
    left_bc.type = femml.BCType.FIXED
    left_bc.nodes = left_nodes
    left_bc.component = -1
    solver.add_boundary_condition(left_bc)

    # Prescribed X-displacement on the right end
    right_bc = femml.BoundaryCondition()
    right_bc.type = femml.BCType.DISPLACEMENT
    right_bc.nodes = right_nodes
    right_bc.component = 0  # X direction
    right_bc.value = target_disp
    right_bc.ramp_time = ramp_time
    solver.add_boundary_condition(right_bc)

    print("   Boundary conditions:")
    print(f"     - Left nodes fixed (IDs {left_nodes})")
    print(
        "     - Right nodes reach "
        f"{target_disp * 1e3:.5f} mm in {ramp_time * 1e3:.1f} ms "
        f"({disp_rate * 1e3:.2f} mm/s)"
    )

    # Time stepping (2 ms total, 0.5 µs step => 4000 steps)
    total_time = ramp_time
    time_step = 5e-7
    num_steps = int(math.ceil(total_time / time_step))

    params = femml.SolverParams()
    params.time_step = time_step
    params.num_steps = num_steps
    params.damping = 0.0
    params.output_interval = max(1, num_steps // 20)
    params.auto_time_step = False
    solver.set_parameters(params)

    print("\n4. Initializing solver...")
    solver.initialize()

    time_history = []
    disp_history = []

    def output_callback(step: int, sim_time: float) -> None:
        if step == 0:
            return
        avg_disp = np.mean([solver.get_node_displacement(n)[0] for n in right_nodes])
        time_history.append(sim_time)
        disp_history.append(avg_disp)

    solver.set_output_callback(output_callback)

    print("\n5. Running analysis...")
    solver.solve()

    print("\n6. Writing results...")
    results_file = "simple_cube_results.csv"
    solver.write_results(results_file)
    print(f"   Saved nodal results -> {results_file}")

    print("\n7. Plotting results...")
    time_ms = np.array(time_history) * 1e3
    disp_mm = np.array(disp_history) * 1e3
    plt.figure(figsize=(8, 5))
    plt.plot(time_ms, disp_mm, lw=2.0, color="navy")
    plt.xlabel("Time (ms)")
    plt.ylabel("Average X Displacement (mm)")
    plt.title("Right Face Prescribed Displacement History")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("displacement_history.png", dpi=150)
    print("   Saved plot: displacement_history.png")

    print("\n8. Final X-displacements (mm):")
    for label, nodes in [
        ("Left (fixed)", left_nodes),
        ("Right (loaded)", right_nodes),
    ]:
        values = [solver.get_node_displacement(n)[0] * 1e3 for n in nodes]
        formatted = ", ".join(f"Node {nid}: {val:.5f}" for nid, val in zip(nodes, values))
        print(f"   {label}: {formatted}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
