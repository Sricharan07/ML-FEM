#!/usr/bin/env python3
"""
Job-1 GPU-Accelerated Simulation
=================================

Runs Job-1 mesh on GPU with 1-second total simulation time.

Features:
- 29,105 nodes, 21,178 elements
- GPU-accelerated explicit dynamics
- Aluminum material
- 1-second simulation time
- Real-time progress monitoring
"""

import sys
import os
import time
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
    print("=" * 70)
    print("ERROR: femml module not found!")
    print("Please build the project with GPU support first:")
    print("  cd FEM-ML")
    print("  .\\build.ps1")
    print("=" * 70)
    sys.exit(1)

# Check if GPU solver is available
if not hasattr(femml, 'GPUExplicitSolver'):
    print("=" * 70)
    print("ERROR: GPU solver not available!")
    print("Please rebuild with CUDA support:")
    print("  cmake -DFEMML_USE_GPU=ON ..")
    print("=" * 70)
    sys.exit(1)

try:
    import pyvista as pv
    HAS_PYVISTA = True
    pv.global_theme.window_size = [1024, 768]
except ImportError:
    HAS_PYVISTA = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

SNAPSHOT_INTERVAL = int(os.environ.get("FEMML_SNAPSHOT_INTERVAL", "1000"))
SNAPSHOT_DIR = Path(os.environ.get("FEMML_SNAPSHOT_DIR", "job1_snapshots"))
FIELD_SELECTION = [
    field.strip().lower() for field in os.environ.get(
        "FEMML_FIELDS", "displacement,stress_vm,strain_vm"
    ).split(",") if field.strip()
]


def von_mises(values):
    sxx = values[:, 0]
    syy = values[:, 1]
    szz = values[:, 2]
    sxy = values[:, 3]
    sxz = values[:, 4]
    syz = values[:, 5]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    )


FIELD_CONFIG = {
    "displacement": {
        "label": "|u| (m)",
        "source": "point",
        "cmap": "viridis",
        "generator": lambda snap: np.linalg.norm(snap["displacements"], axis=1)
    },
    "stress_vm": {
        "label": "Von Mises Stress (Pa)",
        "source": "cell",
        "cmap": "inferno",
        "generator": lambda snap: von_mises(snap["cell_stress"]) if snap["cell_stress"] is not None else None
    },
    "strain_vm": {
        "label": "Von Mises Strain",
        "source": "cell",
        "cmap": "plasma",
        "generator": lambda snap: von_mises(snap["cell_strain"]) if snap["cell_strain"] is not None else None
    },
    "stress_sxx": {
        "label": "Sxx (Pa)",
        "source": "cell",
        "cmap": "coolwarm",
        "generator": lambda snap: snap["cell_stress"][:, 0] if snap["cell_stress"] is not None else None
    },
    "strain_exx": {
        "label": "Exx",
        "source": "cell",
        "cmap": "coolwarm",
        "generator": lambda snap: snap["cell_strain"][:, 0] if snap["cell_strain"] is not None else None
    },
}

ENABLED_FIELDS = [field for field in FIELD_SELECTION if field in FIELD_CONFIG]
if not ENABLED_FIELDS:
    ENABLED_FIELDS = ["displacement"]


def build_pyvista_grid(mesh):
    if not HAS_PYVISTA:
        return None, None

    points = np.array([
        [
            mesh.get_node(node_id).coords[0],
            mesh.get_node(node_id).coords[1],
            mesh.get_node(node_id).coords[2],
        ]
        for node_id in range(1, mesh.get_num_nodes() + 1)
    ])

    cell_data = []
    cell_types = []
    element_ids = []
    for elem_id in range(1, mesh.get_num_elements() + 1):
        elem = mesh.get_element(elem_id)
        node_ids = [nid - 1 for nid in list(elem.nodes)]
        elem_type = getattr(elem, "type", "")

        if elem_type in ("C3D8", "C3D8R") and len(node_ids) == 8:
            cell_data.extend([8, *node_ids])
            cell_types.append(pv.CellType.HEXAHEDRON)
            element_ids.append(elem.id)
        elif elem_type == "C3D4" and len(node_ids) == 4:
            cell_data.extend([4, *node_ids])
            cell_types.append(pv.CellType.TETRA)
            element_ids.append(elem.id)

    if not cell_types:
        return None, None

    cells = np.array(cell_data, dtype=np.int64)
    types = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, points)
    return grid, element_ids


def save_snapshot_images(grid, snapshots, output_dir):
    if not HAS_PYVISTA or grid is None or not snapshots:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_points = grid.points.copy()
    summary = {}

    for field in ENABLED_FIELDS:
        cfg = FIELD_CONFIG.get(field)
        if cfg is None:
            continue

        field_dir = output_dir / field
        field_dir.mkdir(parents=True, exist_ok=True)
        images = []

        for snapshot in snapshots:
            scalars = cfg["generator"](snapshot)
            if scalars is None:
                continue

            frame = grid.copy()
            frame.points = base_points + snapshot["displacements"]

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(
                frame,
                scalars=scalars,
                cmap=cfg.get("cmap", "viridis"),
                scalar_bar_args={"title": cfg["label"], "vertical": True},
                show_edges=False,
            )
            plotter.view_isometric()
            plotter.add_text(
                f"Step {snapshot['step']:,}  Time {snapshot['time']:.6f}s",
                font_size=10,
            )
            plotter.add_axes(color="black")
            plotter.show_grid(color="lightgrey")

            filename = field_dir / f"{field}_{snapshot['step']:06d}.png"
            plotter.show(screenshot=str(filename))
            plotter.close()
            images.append(filename)

        if HAS_IMAGEIO and len(images) > 1:
            gif_path = field_dir / f"{field}.gif"
            imageio.mimsave(
                gif_path,
                [imageio.imread(str(path)) for path in images],
                duration=0.25,
            )
            images.append(gif_path)

        summary[field] = images

    return summary


def main():
    print("=" * 70)
    print("  FEM-ML GPU: Job-1 Simulation")
    print("=" * 70)

    # ========================================================================
    # 1. Load Mesh
    # ========================================================================
    print("\n[1/7] Loading Job-1 mesh...")
    start_time = time.time()

    importer = femml.AbaqusImporter()
    job1_path = os.path.join(_HERE, "Job-1.inp")

    if not os.path.exists(job1_path):
        print(f"ERROR: Job-1.inp not found at {job1_path}")
        sys.exit(1)

    mesh = importer.import_mesh(job1_path)

    num_nodes = mesh.get_num_nodes()
    num_elements = mesh.get_num_elements()

    load_time = time.time() - start_time
    print(f"   ✓ Loaded mesh: {num_nodes:,} nodes, {num_elements:,} elements")
    print(f"   ✓ Load time: {load_time:.2f} s")

    # Gather node information for boundary detection
    node_list = []
    for node_id in range(1, num_nodes + 1):
        try:
            node = mesh.get_node(node_id)
            node_list.append(node)
        except Exception:
            continue

    x_coords = [n.coords[0] for n in node_list]
    x_min = min(x_coords)
    x_max = max(x_coords)
    tol = max(1e-6, 1e-3 * (x_max - x_min))
    left_nodes = [n.id for n in node_list if n.coords[0] <= x_min + tol]
    right_nodes = [n.id for n in node_list if n.coords[0] >= x_max - tol]
    if not left_nodes or not right_nodes:
        print("Warning: Failed to detect boundary node sets automatically; reverting to default ranges.")
        left_nodes = list(range(1, min(101, num_nodes + 1)))
        right_nodes = list(range(max(1, num_nodes - 99), num_nodes + 1))

    pv_grid, cell_elem_ids = build_pyvista_grid(mesh) if HAS_PYVISTA else (None, None)
    capture_snapshots = HAS_PYVISTA and pv_grid is not None and cell_elem_ids is not None and SNAPSHOT_INTERVAL > 0
    snapshots = []

    # ========================================================================
    # 2. Create Material
    # ========================================================================
    print("\n[2/7] Creating material...")
    material = femml.create_aluminum()
    print(f"   ✓ Material: {material.get_name()}")
    print(f"   ✓ Young's modulus: {material.get_youngs_modulus():.2e} Pa")
    print(f"   ✓ Poisson's ratio: {material.get_poissons_ratio():.3f}")
    print(f"   ✓ Density: {material.get_density():.1f} kg/m³")

    # ========================================================================
    # 3. Create GPU Solver
    # ========================================================================
    print("\n[3/7] Initializing GPU solver...")
    solver = femml.GPUExplicitSolver(mesh)
    solver.set_material(material)

    # Set GPU device (use device 0 by default)
    solver.set_device_id(0)
    print(f"   ✓ Using GPU device: {solver.get_device_id()}")

    # ========================================================================
    # 4. Apply Boundary Conditions
    # ========================================================================
    print("\n[4/7] Setting boundary conditions...")

    # Fix left end nodes (assume nodes 1-100 are on left boundary)
    # In a real scenario, you would identify these from the mesh
    bc = femml.BoundaryCondition()
    bc.type = femml.BCType.FIXED
    bc.nodes = left_nodes
    bc.component = -1  # All DOFs
    solver.add_boundary_condition(bc)
    print(f"   ✓ Fixed {len(bc.nodes)} nodes on left boundary (x ≈ {x_min:.3f})")

    # ========================================================================
    # 5. Apply Loads
    # ========================================================================
    print("\n[5/7] Applying loads...")

    # Apply tensile load on right end nodes (last 100 nodes)
    force_magnitude = 1e6  # 1 MN
    load = femml.Load()
    load.type = femml.LoadType.FORCE
    load.nodes = right_nodes
    load.component = 0  # X direction (tensile)
    load.value = force_magnitude / len(load.nodes)  # Distribute load
    solver.add_load(load)
    print(f"   ✓ Applied {force_magnitude:.2e} N tensile load")
    print(f"   ✓ Distributed over {len(load.nodes)} nodes on right boundary (x ≈ {x_max:.3f})")

    # ========================================================================
    # 6. Configure Solver Parameters
    # ========================================================================
    print("\n[6/7] Configuring solver...")

    params = femml.SolverParams()
    params.auto_time_step = False
    params.time_step_scale = 0.8  # Safety factor
    params.num_steps = 0  # Will be calculated from total time
    params.damping = 0.02  # 2% damping
    params.output_interval = 100

    # Calculate number of steps for 1 second simulation
    # Estimate critical time step first
    material_wave_speed = np.sqrt(material.get_youngs_modulus() / material.get_density())
    element_size = 1.0  # Approximate element size in meters (adjust if known)
    estimated_dt = element_size / material_wave_speed * params.time_step_scale

    total_time = 1.0  # 1 second
    params.num_steps = int(total_time / estimated_dt)
    params.time_step = estimated_dt

    print(f"   ✓ Total simulation time: {total_time} s")
    print(f"   ✓ Estimated time step: {estimated_dt:.2e} s")
    print(f"   ✓ Number of steps: {params.num_steps:,}")
    print(f"   ✓ Damping: {params.damping * 100:.1f}%")

    if capture_snapshots:
        params.output_interval = max(1, min(params.output_interval, SNAPSHOT_INTERVAL))
    solver.set_parameters(params)
    if capture_snapshots:
        print(f"   ✓ Snapshot interval: every {SNAPSHOT_INTERVAL} steps (saved in {SNAPSHOT_DIR})")
    elif SNAPSHOT_INTERVAL > 0 and not HAS_PYVISTA:
        print("   ✓ Contour snapshots disabled (PyVista not available)")

    def capture_snapshot(step_idx, sim_time):
        if not capture_snapshots:
            return
        disp = solver.get_displacements()
        disp_array = np.array([[d[0], d[1], d[2]] for d in disp])
        cell_stress = None
        cell_strain = None
        try:
            elem_results = {res.id: res for res in solver.get_element_results()}
            cell_stress = []
            cell_strain = []
            for elem_id in cell_elem_ids:
                res = elem_results.get(elem_id)
                if res is None:
                    cell_stress = None
                    cell_strain = None
                    break
                cell_stress.append(list(res.stress))
                cell_strain.append(list(res.strain))
            if cell_stress is not None:
                cell_stress = np.array(cell_stress)
                cell_strain = np.array(cell_strain)
        except Exception:
            cell_stress = None
            cell_strain = None

        snapshots.append({
            "step": step_idx,
            "time": sim_time,
            "displacements": disp_array,
            "cell_stress": cell_stress,
            "cell_strain": cell_strain,
        })

    # ========================================================================
    # 7. Run Simulation
    # ========================================================================
    print("\n[7/7] Running GPU simulation...")
    print("   " + "-" * 60)

    # Progress callback
    last_update_time = [time.time()]

    def progress_callback(step, sim_time):
        if time.time() - last_update_time[0] > 1.0:  # Update every second
            progress = 100.0 * step / params.num_steps
            ke = solver.get_kinetic_energy()
            se = solver.get_strain_energy()

            print(f"   Step {step:8,}/{params.num_steps:,} ({progress:5.1f}%) | "
                  f"Time: {sim_time:.4f} s | KE: {ke:.3e} J | SE: {se:.3e} J")

            last_update_time[0] = time.time()
        if capture_snapshots and (step % SNAPSHOT_INTERVAL == 0 or step == params.num_steps):
            capture_snapshot(step, sim_time)

    solver.set_output_callback(progress_callback)

    # Initialize solver (allocates GPU memory, computes mass matrix)
    init_start = time.time()
    solver.initialize()
    init_time = time.time() - init_start
    print(f"   ✓ Initialization time: {init_time:.2f} s")
    print(f"   ✓ GPU memory usage: {solver.get_gpu_memory_usage() / (1024**2):.1f} MB")
    print()
    if capture_snapshots:
        capture_snapshot(0, 0.0)

    # Solve
    solve_start = time.time()
    solver.solve()
    solve_time = time.time() - solve_start
    if capture_snapshots and (not snapshots or snapshots[-1]["step"] != params.num_steps):
        capture_snapshot(params.num_steps, solver.get_current_time())

    print()
    print("   " + "-" * 60)
    print(f"   ✓ Simulation complete!")
    print(f"   ✓ Total solve time: {solve_time:.2f} s")
    print(f"   ✓ GPU compute time: {solver.get_gpu_compute_time():.2f} s")
    print(f"   ✓ CPU-GPU transfer time: {solver.get_cpu_gpu_transfer_time():.2f} s")
    print(f"   ✓ Performance: {params.num_steps / solve_time:.1f} steps/s")

    # ========================================================================
    # 8. Results
    # ========================================================================
    print("\n[8/8] Processing results...")

    displacements = solver.get_displacements()
    disp_array = np.array([[d[0], d[1], d[2]] for d in displacements])
    disp_mag = np.linalg.norm(disp_array, axis=1)

    print(f"   ✓ Maximum displacement: {np.max(disp_mag) * 1000:.4f} mm")
    print(f"   ✓ Average displacement: {np.mean(disp_mag) * 1000:.4f} mm")

    # Final energies
    ke_final = solver.get_kinetic_energy()
    se_final = solver.get_strain_energy()
    print(f"   ✓ Final kinetic energy: {ke_final:.6e} J")
    print(f"   ✓ Final strain energy: {se_final:.6e} J")

    # ========================================================================
    # 9. Export Results
    # ========================================================================
    print("\n[9/9] Exporting results...")

    output_csv = "job1_results.csv"
    solver.write_results(output_csv)
    print(f"   ✓ Saved: {output_csv}")

    # Save displacement magnitude for visualization
    np.savetxt("job1_displacement_magnitude.txt", disp_mag)
    print(f"   ✓ Saved: job1_displacement_magnitude.txt")

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Displacement histogram
    ax1.hist(disp_mag * 1000, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Displacement Magnitude (mm)')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title('Displacement Distribution')
    ax1.grid(True, alpha=0.3)

    # Displacement along node index (simplified visualization)
    ax2.plot(disp_mag * 1000, linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Node Index')
    ax2.set_ylabel('Displacement Magnitude (mm)')
    ax2.set_title('Displacement Profile')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = "job1_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {plot_file}")

    # ========================================================================
    # 10. Contour visualization
    # ========================================================================
    contour_summary = None
    if capture_snapshots and snapshots:
        print("\n[10/10] Generating contour plots...")
        field_images = save_snapshot_images(pv_grid, snapshots, SNAPSHOT_DIR)
        if any(field_images.values()):
            contour_summary = field_images
            for field, paths in field_images.items():
                png_count = sum(1 for p in paths if p.suffix.lower() == ".png")
                if png_count == 0:
                    continue
                print(f"   ✓ {field} : {png_count} frames -> {SNAPSHOT_DIR / field}")
        else:
            print("   ✓ Snapshot data available but no images were produced.")
    elif SNAPSHOT_INTERVAL > 0 and not HAS_PYVISTA:
        print("\n[10/10] Contour plots skipped (PyVista not installed).")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Mesh: {num_nodes:,} nodes, {num_elements:,} elements")
    print(f"  Material: {material.get_name()}")
    print(f"  Total time: {total_time} s ({params.num_steps:,} steps)")
    print(f"  Time step per increment: {params.time_step:.6e} s")
    print(f"  Solve time: {solve_time:.2f} s ({params.num_steps / solve_time:.1f} steps/s)")
    print(f"  GPU speedup: {solve_time / solver.get_gpu_compute_time():.1f}x")
    print(f"  Max displacement: {np.max(disp_mag) * 1000:.4f} mm")
    if contour_summary:
        print("  Contour outputs:")
        for field, paths in contour_summary.items():
            png_count = sum(1 for p in paths if p.suffix.lower() == ".png")
            if png_count:
                print(f"    - {field}: {png_count} PNG frames in {SNAPSHOT_DIR / field}")
            gif_paths = [p for p in paths if p.suffix.lower() == ".gif"]
            for gif in gif_paths:
                print(f"      GIF: {gif}")
    print("=" * 70)
    print("\n✓ Job-1 GPU simulation complete!\n")


if __name__ == "__main__":
    main()
