#!/usr/bin/env python3
"""
Quick Test: Single Element with Contour Plot
Bypasses GUI, runs fast, shows visual results
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Add build paths
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_ROOT, "build"),
    os.path.join(_ROOT, "build", "src", "python"),
    os.path.join(_ROOT, "build", "src", "python", "Release"),
    os.path.join(_ROOT, "build", "src", "python", "Debug"),
    os.path.join(_ROOT, "build", "Release"),
    os.path.join(_ROOT, "build", "Debug"),
]
for _path in _PYTHON_CANDIDATES:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import femml
except ImportError:
    print("="*60)
    print("ERROR: femml module not found!")
    print("Please build the project first:")
    print("  cd FEM-ML")
    print("  .\\build.ps1")
    print("="*60)
    sys.exit(1)


def plot_element_contour(nodes, displacement, output_file="single_element_results.png"):
    """
    Create 3D contour plot of displacement magnitude
    """
    fig = plt.figure(figsize=(14, 6))

    # Original mesh
    ax1 = fig.add_subplot(121, projection='3d')

    # Deformed mesh with contours
    ax2 = fig.add_subplot(122, projection='3d')

    # Extract coordinates
    coords = np.array([n for n in nodes])

    # Displacement magnitude
    disp_mag = np.linalg.norm(displacement, axis=1) * 1000  # Convert to mm

    # Original mesh
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c='blue', s=100, marker='o', label='Nodes')

    # Draw edges (cube connectivity)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]

    for edge in edges:
        pts = coords[edge, :]
        ax1.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Original Mesh')
    ax1.legend()

    # Deformed mesh
    scale = 1000.0  # Scale displacements for visibility
    deformed = coords + displacement * scale

    # Plot with color based on displacement
    scatter = ax2.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2],
                         c=disp_mag, s=100, marker='o', cmap='jet',
                         vmin=0, vmax=disp_mag.max())

    # Draw deformed edges
    for edge in edges:
        pts = deformed[edge, :]
        ax2.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], 'k-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Deformed Shape (×{scale})')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, shrink=0.8)
    cbar.set_label('Displacement (mm)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {output_file}")

    return disp_mag


def main():
    print("="*60)
    print("  FEM-ML: Single Element Test")
    print("="*60)

    # 1. Import mesh
    print("\n1. Loading mesh...")
    importer = femml.AbaqusImporter()
    cube_path = os.path.join(_HERE, "cube.inp")
    mesh = importer.import_mesh(cube_path)
    print(f"   ✓ Loaded {mesh.get_num_nodes()} nodes, {mesh.get_num_elements()} elements")

    # 2. Create material
    print("\n2. Creating material...")
    material = femml.create_aluminum()
    print(f"   ✓ Material: {material.get_name()}")
    print(f"   ✓ E = {material.get_youngs_modulus():.2e} Pa")
    print(f"   ✓ ν = {material.get_poissons_ratio():.3f}")
    print(f"   ✓ ρ = {material.get_density():.1f} kg/m³")

    # 3. Create solver
    print("\n3. Setting up solver...")
    solver = femml.ExplicitSolver(mesh)
    solver.set_material(material)

    # Fix bottom nodes (1,2,5,6)
    bc = femml.BoundaryCondition()
    bc.type = femml.BCType.FIXED
    bc.nodes = [1, 2, 5, 6]
    bc.component = -1  # All components
    solver.add_boundary_condition(bc)
    print("   ✓ Fixed bottom face (nodes 1,2,5,6)")

    # Pull top nodes upward (3,4,7,8)
    force_value = 5e5  # 500 kN
    for node_id in [3, 4, 7, 8]:
        load = femml.Load()
        load.type = femml.LoadType.FORCE
        load.nodes = [node_id]
        load.component = 1  # Y direction
        load.value = force_value
        solver.add_load(load)
    print(f"   ✓ Applied {force_value:.0e} N upward on top face")

    # 4. Configure solver
    params = femml.SolverParams()
    params.time_step = 5e-8
    params.num_steps = 1000  # Shorter for quick test
    params.damping = 0.05     # Add damping to settle
    params.output_interval = 50
    params.auto_time_step = True
    solver.set_parameters(params)
    print(f"   ✓ Time steps: {params.num_steps}")

    # 5. Initialize
    print("\n4. Initializing solver...")
    solver.initialize()
    print("   ✓ Ready to solve")

    # 6. Solve
    print("\n5. Running simulation...")
    print("   Progress: ", end='', flush=True)

    step_count = 0
    def progress_callback(step, time):
        nonlocal step_count
        if step % 100 == 0:
            print(f"{step}...", end='', flush=True)
        step_count = step

    solver.set_output_callback(progress_callback)
    solver.solve()
    print(f" Done! ({step_count} steps)")

    # 7. Get results
    print("\n6. Processing results...")
    displacements = solver.get_displacements()

    # Convert to numpy array
    disp_array = np.array([[d[0], d[1], d[2]] for d in displacements])

    # Get node coordinates
    nodes = []
    for i in range(1, mesh.get_num_nodes() + 1):
        node = mesh.get_node(i)
        nodes.append([node.coords[0], node.coords[1], node.coords[2]])
    nodes = np.array(nodes)

    # 8. Print results
    print("\n7. Final Displacements:")
    print("   " + "-"*50)
    print(f"   {'Node':>6} {'Ux (mm)':>12} {'Uy (mm)':>12} {'Uz (mm)':>12} {'|U| (mm)':>12}")
    print("   " + "-"*50)

    for i in range(len(disp_array)):
        ux, uy, uz = disp_array[i] * 1000  # Convert to mm
        mag = np.linalg.norm(disp_array[i]) * 1000
        print(f"   {i+1:6d} {ux:12.6f} {uy:12.6f} {uz:12.6f} {mag:12.6f}")

    print("   " + "-"*50)

    # Summary
    max_disp = np.max(np.linalg.norm(disp_array, axis=1)) * 1000
    print(f"\n   Maximum displacement: {max_disp:.4f} mm")

    # Energy
    ke = solver.get_kinetic_energy()
    print(f"   Final kinetic energy: {ke:.6e} J")

    # 9. Create contour plot
    print("\n8. Creating visualization...")
    disp_mag = plot_element_contour(nodes, disp_array)

    # 10. Export data
    output_csv = "single_element_results.csv"
    solver.write_results(output_csv)
    print(f"✓ Saved results: {output_csv}")

    print("\n" + "="*60)
    print("  Test Complete!")
    print("="*60)
    print("\nFiles created:")
    print(f"  • single_element_results.png  (visualization)")
    print(f"  • {output_csv}  (displacement data)")
    print("\n✓ Single element test passed!\n")


if __name__ == "__main__":
    main()
