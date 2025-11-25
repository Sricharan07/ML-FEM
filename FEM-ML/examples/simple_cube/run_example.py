#!/usr/bin/env python3
"""
Simple example demonstrating FEM-ML Python API
"""

import sys
import os

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
    import femml
except ImportError:
    print("Error: femml module not found. Please build the project first.")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("FEM-ML Simple Cube Example")
    print("=" * 60)

    # Import mesh
    print("\n1. Importing mesh...")
    importer = femml.AbaqusImporter()
    cube_path = os.path.join(_HERE, "cube.inp")
    mesh = importer.import_mesh(cube_path)

    # Create material
    print("\n2. Creating material...")
    material = femml.create_aluminum()
    print(f"   Material: {material.get_name()}")
    print(f"   Density: {material.get_density()} kg/m³")

    # Create solver
    print("\n3. Setting up solver...")
    solver = femml.ExplicitSolver(mesh)

    # Assign material
    solver.set_material(material)

    # Add boundary conditions (fix bottom face)
    bc = femml.BoundaryCondition()
    bc.type = femml.BCType.FIXED
    bc.nodes = [1, 2, 5, 6]
    bc.component = -1  # All components
    solver.add_boundary_condition(bc)

    # Add load (pull top face in Y direction)
    for node_id in [3, 4, 7, 8]:
        load = femml.Load()
        load.type = femml.LoadType.FORCE
        load.nodes = [node_id]
        load.component = 1  # Y direction
        load.value = 5e5  # 500 kN
        solver.add_load(load)

    # Set solver parameters
    params = femml.SolverParams()
    params.time_step = 5e-8
    params.num_steps = 2000
    params.damping = 0.0
    params.output_interval = 50
    params.auto_time_step = True
    solver.set_parameters(params)

    # Initialize solver
    print("\n4. Initializing solver...")
    solver.initialize()

    # Track displacement history
    time_history = []
    disp_history = []

    def output_callback(step, time):
        # Track node 8 displacement
        disp = solver.get_node_displacement(8)
        time_history.append(time)
        disp_history.append(disp[1])  # Y displacement

    solver.set_output_callback(output_callback)

    # Solve
    print("\n5. Running analysis...")
    solver.solve()

    # Write results
    print("\n6. Writing results...")
    solver.write_results("results.csv")

    # Plot displacement history
    print("\n7. Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(time_history) * 1e6, np.array(disp_history) * 1e3)
    plt.xlabel("Time (μs)")
    plt.ylabel("Y Displacement (mm)")
    plt.title("Node 8 Displacement History")
    plt.grid(True)
    plt.savefig("displacement_history.png", dpi=150)
    print("   Saved plot: displacement_history.png")

    # Print final displacements
    print("\n8. Final displacements:")
    for node_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        disp = solver.get_node_displacement(node_id)
        print(f"   Node {node_id}: "
              f"({disp[0]*1e3:.4f}, {disp[1]*1e3:.4f}, {disp[2]*1e3:.4f}) mm")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
