# ML-FEM-Pytorch

Minimal PyTorch-based explicit FEM solver paired with a lightweight GUI and Abaqus mesh importer ported from the original FEM-ML project.

## Features

- Fully vectorized explicit dynamics solver implemented in `logic.py` (mass scaling, lumped mass assembly, hourglass stabilization, energy tracking).
- Python-native Abaqus `.inp` importer (`femml/mesh.py`) with mesh, node sets, element sets, and surface data structures for reuse outside the GUI.
- PyQt5/PyVista GUI (`gui/main_window.py`) for importing meshes, defining boundary conditions, running jobs, and previewing deformed shapes in real time.
- Maintains a minimal directory footprint: only `logic.py`, `femml/`, and `gui/`.

## Requirements

Install dependencies in your Python environment (3.9+ recommended):

```bash
pip install torch numpy scipy matplotlib pyqt5 pyvista pyvistaqt
```

If you only need the solver without the GUI, you can skip the Qt/PyVista packages.

## Directory Layout

| Path                          | Description |
|------------------------------|-------------|
| `logic.py`                   | Core explicit solver, structured mesh helpers, and notch demo. |
| `femml/mesh.py`              | Port of FEM-ML mesh + Abaqus importer (pure Python). |
| `gui/main_window.py`         | PyQt5 front-end with mesh tree, BC editor, solver controls, and viewer. |
| `gui/__init__.py`, `femml/__init__.py` | Empty files so modules can be imported via `python -m`. |


## Quick Start: CLI / Notebook

Run the included demo (structured plate with a semicircular notch) which produces the same plots as the original script:

```bash
cd ML-FEM-Pytorch
python logic.py
```

To use the solver programmatically:

```python
from logic import (
    create_structured_mesh,
    PyTorchExplicitSolver,
    SimulationConfig,
    MaterialProperties,
    BoundaryCondition,
)
import numpy as np

coords, elements = create_structured_mesh(nx=10, ny=4, nz=1, lengths=(100.0, 20.0, 1.0))
config = SimulationConfig(total_time=0.01, target_time_step=5e-6)
material = MaterialProperties(density=2.7e-9, youngs_modulus=70_000.0, poisson_ratio=0.32)

x_vals = coords[:, 0].cpu().numpy()
left_nodes = np.where(np.isclose(x_vals, 0.0))[0].tolist()
right_nodes = np.where(np.isclose(x_vals, 100.0))[0].tolist()

bcs = [
    BoundaryCondition(name="Clamp-Left", node_indices=left_nodes, components=(-1,), bc_type="fixed"),
    BoundaryCondition(name="Disp-Right-X", node_indices=right_nodes, components=(0,), bc_type="displacement", value=0.5),
]

solver = PyTorchExplicitSolver(coords, elements, material=material, config=config)
result = solver.run(bcs)
print("Final time:", result.time_history[-1], "Max |u|:", result.displacements.abs().max().item())
```

`SimulationResults` stores tensors for displacements, velocities, element stresses, nodal stresses, and energy histories, so you can post-process them in NumPy or torch.

## Quick Start: GUI

1. **Launch**

   ```bash
   cd ML-FEM-Pytorch
   python -m gui.main_window
   ```

2. **Import Mesh** – Click *Import Mesh* and load an Abaqus `.inp`. The importer builds node/element sets for BC dialogs.

3. **Define BCs** – Use *Add BC* to select node sets or enter ranges manually (`1-10, 25, 30`). Choose components (All/X/Y/Z) and specify displacement magnitude + ramp time.

4. **Configure Solver** – Adjust total time, target time step, density, material properties, and device (Auto/CPU/GPU).

5. **Run & View** – Press *Run Simulation*. Progress updates stream into the log and the PyVista viewport animates the deformed shape (scalar field by σₓₓ or displacement magnitude).

## Using the Abaqus Importer

```python
from femml.mesh import AbaqusInpImporter

importer = AbaqusInpImporter(verbose=True)
mesh = importer.import_file("path/to/model.inp")
coords, elements, node_ids, element_ids = mesh.to_numpy()
node_set_indices = mesh.node_indices_from_set("CLAMP_LEFT")
```

Use `mesh.summary()` for counts and `mesh.bounding_box()` for extents. The importer handles `*NODE`, `*ELEMENT`, `*NSET`, `*ELSET`, and `*SURFACE` sections, including generated sets.

## Customization Tips

- **Material Models**: The solver currently assumes isotropic linear elasticity. Update `MaterialProperties` and `_build_constitutive_matrix()` if you need different constitutive laws.
- **Mass Scaling / Damping**: Tune `SimulationConfig` (`target_time_step`, `max_mass_scale`, `bulk_alpha`, `bulk_beta`, `hourglass_gamma`) for stability vs. accuracy trade-offs.
- **Boundary Conditions**: BCs are Dirichlet-only (fixed/displacement). Extend `_apply_dirichlet` if you need velocity ramps or other prescribed quantities.
- **Visualization**: The GUI deformation scale is adjustable; for scripting, call `plot_notched_plate` or implement your own matplotlib/PyVista workflow using the tensors in `SimulationResults`.

## Troubleshooting

- **PyVista errors**: Install `pyvistaqt` and ensure an OpenGL-compatible GPU/driver is available. Running headless? Launch with the `PYVISTA_OFF_SCREEN=true` environment variable and skip the GUI view.
- **No GPU in torch**: The solver automatically falls back to CPU if `torch.cuda.is_available()` returns `False`.
- **Large meshes**: Consider reducing output frequency or running the solver headless for large `.inp` models; the GUI runs the solver in a background thread but still stores full displacement/stress tensors.

---

For additional ideas, inspect `logic.py` (demo at the bottom) and `gui/main_window.py` (PyQt5 integration). Both files are self-contained and meant to be edited directly. Happy hacking!
