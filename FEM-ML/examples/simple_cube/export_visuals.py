#!/usr/bin/env python3
"""Generate publication-style figures for the simple cube example."""

import os
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Ensure femml module is importable (same search logic as run_example.py)
PYTHON_CANDIDATES = [
    ROOT / "build",
    ROOT / "build" / "src" / "python",
    ROOT / "build" / "src" / "python" / "Release",
    ROOT / "build" / "src" / "python" / "Debug",
]
for candidate in PYTHON_CANDIDATES:
    candidate = str(candidate)
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import femml
except ImportError:  # pragma: no cover - easier failure message
    raise SystemExit(
        "Unable to import femml. Build the project first (see README.md)."
    ) from None


def set_axes_equal(ax: plt.Axes, reference_nodes: np.ndarray = None) -> None:
    """Force equal aspect ratio across all axes for 3D plots.

    Args:
        ax: The matplotlib 3D axes to adjust
        reference_nodes: If provided, use these coordinates to determine limits
                        instead of the current axis limits. This ensures all
                        plots have the same bounds.
    """
    if reference_nodes is not None:
        # Use reference nodes to set consistent limits across all plots
        mins = reference_nodes.min(axis=0)
        maxs = reference_nodes.max(axis=0)
        limits = np.column_stack([mins, maxs])
    else:
        # Fall back to current axis limits
        limits = np.array(
            [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float
        )

    span = limits[:, 1] - limits[:, 0]
    center = np.mean(limits, axis=1)
    max_span = np.max(span)

    # Add 10% padding for better visualization
    max_span *= 1.1

    new_limits = np.column_stack([center - max_span / 2.0, center + max_span / 2.0])
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])


def hexa_edges():
    return [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


def hexa_faces():
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]


def create_axes():
    """Create a Matplotlib 3D axis with consistent styling."""
    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    # Consistent viewing angle for all plots
    ax.view_init(elev=28, azim=-60)
    # Labels
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    # Initial limits (will be adjusted by set_axes_equal)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    return fig, ax


def run_simple_cube():
    """Solve the single-element cube and return field data."""
    importer = femml.AbaqusImporter()
    mesh = importer.import_mesh(str(HERE / "cube.inp"))

    solver = femml.ExplicitSolver(mesh)
    solver.set_material(femml.create_aluminum())

    # Bottom fixed
    bottom_bc = femml.BoundaryCondition()
    bottom_bc.type = femml.BCType.FIXED
    bottom_bc.nodes = [1, 2, 5, 6]
    bottom_bc.component = -1
    solver.add_boundary_condition(bottom_bc)

    # Top prescribed displacement (0.5 mm in Y over 0.05 s)
    top_bc = femml.BoundaryCondition()
    top_bc.type = femml.BCType.DISPLACEMENT
    top_bc.nodes = [3, 4, 7, 8]
    top_bc.component = 1
    top_bc.value = 5e-4
    top_bc.ramp_time = 0.05
    solver.add_boundary_condition(top_bc)

    params = femml.SolverParams()
    params.time_step = 1e-6
    params.num_steps = 200
    params.output_interval = 20
    params.damping = 0.0
    params.auto_time_step = True
    solver.set_parameters(params)

    solver.initialize()
    solver.solve()

    num_nodes = mesh.get_num_nodes()
    nodes = np.array([mesh.get_node(i).coords for i in range(1, num_nodes + 1)])
    displacements = np.array(
        [solver.get_node_displacement(i) for i in range(1, num_nodes + 1)]
    )

    elem_results = solver.get_element_results()
    if not elem_results:
        raise RuntimeError("Solver did not return element stresses.")

    stress = np.array(elem_results[0].stress, dtype=float)
    sxx, syy, szz, sxy, sxz, syz = stress
    von_mises = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )
    vm_field = np.full(num_nodes, von_mises)

    return nodes, displacements, vm_field


def plot_wireframe(nodes: np.ndarray, out_path: Path,
                   reference_bounds: np.ndarray = None) -> None:
    """Original mesh wireframe.

    Args:
        nodes: Node coordinates
        out_path: Output file path
        reference_bounds: If provided, coordinates to use for axis bounds (to match other plots)
    """
    fig, ax = create_axes()
    for i, j in hexa_edges():
        pts = np.vstack([nodes[i], nodes[j]])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="royalblue", linewidth=2)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="royalblue", s=50,
               edgecolors='darkblue', linewidth=0.5)
    ax.set_title("Original Mesh (1 Hex Element)", fontsize=12, weight='bold')
    # Use reference bounds if provided, otherwise use nodes
    ref = reference_bounds if reference_bounds is not None else nodes
    set_axes_equal(ax, reference_nodes=ref)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_field(
    nodes: np.ndarray,
    displacements: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap_name: str,
    deformation_scale: float,
    out_path: Path,
    reference_bounds: np.ndarray = None,
) -> None:
    """Render colored cube using Matplotlib with consistent axes.

    Args:
        nodes: Original node coordinates
        displacements: Displacement vectors
        values: Scalar values to color by
        title: Plot title
        colorbar_label: Label for colorbar
        cmap_name: Colormap name
        deformation_scale: Scale factor for displacements
        out_path: Output file path
        reference_bounds: If provided, coordinates to use for axis bounds
    """
    fig, ax = create_axes()
    deformed = nodes + displacements * deformation_scale
    faces = hexa_faces()
    face_polys = [[deformed[idx] for idx in face] for face in faces]
    per_face_vals = np.array([np.mean(values[face]) for face in faces])

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + (1e-9 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    poly = Poly3DCollection(
        face_polys,
        facecolors=cmap(norm(per_face_vals)),
        edgecolor="black",
        linewidth=0.9,
        alpha=0.97,
    )
    ax.add_collection3d(poly)
    ax.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2], s=12, color="k", alpha=0.5)

    ax.set_title(title, fontsize=12, weight='bold')
    # Use reference bounds if provided for consistency, otherwise use deformed coords
    ref = reference_bounds if reference_bounds is not None else deformed
    set_axes_equal(ax, reference_nodes=ref)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(values)
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, fraction=0.06)
    cbar.set_label(colorbar_label, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    print("="*60)
    print("Exporting Simple Cube Visualizations")
    print("="*60)

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)

    print("\n1. Running simulation...")
    nodes, displacements, vm = run_simple_cube()
    deformation_scale = 1000.0  # match GUI default
    print("   [OK] Simulation complete")

    disp_m = np.linalg.norm(displacements, axis=1)  # meters
    vm_pa = vm  # Pa to match GUI

    # IMPORTANT: Use ORIGINAL mesh bounds for axes to show true physical dimensions
    # The deformed mesh (with 1000x scale) will extend beyond these bounds,
    # which correctly shows that the deformation is exaggerated for visualization
    reference_coords = nodes  # Use original, not deformed!

    print("\n2. Generating mesh visualization...")
    plot_wireframe(nodes, out_dir / "simple_cube_mesh.png",
                   reference_bounds=reference_coords)

    print("\n3. Generating displacement visualization...")
    plot_field(
        nodes,
        displacements,
        disp_m,
        title=f"Displacement Magnitude (Deformed x{deformation_scale:.0f})",
        colorbar_label="Displacement Magnitude (m)",
        cmap_name="jet",  # Match GUI colormap
        deformation_scale=deformation_scale,
        out_path=out_dir / "simple_cube_displacement.png",
        reference_bounds=reference_coords,
    )

    print("\n4. Generating stress visualization...")
    plot_field(
        nodes,
        displacements,
        vm_pa,
        title=f"Von Mises Stress (Deformed x{deformation_scale:.0f})",
        colorbar_label="Von Mises Stress (Pa)",
        cmap_name="inferno",  # Match GUI colormap
        deformation_scale=deformation_scale,
        out_path=out_dir / "simple_cube_stress.png",
        reference_bounds=reference_coords,
    )

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nSaved figures to: {out_dir}")
    print("\nAll visualizations now have:")
    print("  [+] Consistent axis limits (based on PHYSICAL dimensions)")
    print("  [+] Axes represent true physical size (0-1 m cube)")
    print("  [+] Deformation exaggerated 1000x for visibility")
    print("  [+] Consistent viewing angle (elev=28, azim=-60)")
    print("  [+] GUI-matching colormaps (jet for displacement, inferno for stress)")
    print("  [+] Same aspect ratio and styling")
    print("\nGenerated files:")
    print("  - simple_cube_mesh.png")
    print("  - simple_cube_displacement.png")
    print("  - simple_cube_stress.png")
    print()


if __name__ == "__main__":
    main()
