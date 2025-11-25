# FEM-ML: Explicit FEM Solver with Neural Network Integration

A high-performance explicit finite element method (FEM) solver with Python API, neural network integration, and an Abaqus CAE-like GUI interface.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)

## Features

### Core Capabilities
- Multi-element explicit dynamics solver
- Support for hexahedral (C3D8/C3D8R) and tetrahedral (C3D4) elements
- Abaqus-style stress-increment integration across all materials
- Linear elastic and neural/ML constitutive models
- Abaqus .inp import, automatic time-step sizing, and energy monitoring

### Python Integration
- Full PyBind11 API that mirrors the C++ core
- NumPy-friendly access to meshes, materials, and solver data
- Scriptable workflows for preprocessing, solving, and exporting
- Neural network integration via user-supplied callbacks or ONNX (future)

### GUI Application
- Abaqus/CAE-like PyQt5 interface with PyVista 3D visualization
- Real-time job monitoring (progress, energy, deformation)
- Model tree for nodes, sets, materials, loads, and jobs

## Architecture

```
GUI (PyQt5/PyVista)
    -> Python API (PyBind11 bindings)
        -> C++ Core (mesh, elements, materials, solvers, IO)
```



```
┌─────────────────────────────────────────────────┐
│         GUI (Python/Qt)                         │
│  - 3D Visualization                             │
│  - Job Management                               │
│  - Model Tree                                   │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────┐
│         Python API (PyBind11)                   │
│  - Mesh, Material, Solver bindings              │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────┐
│         C++ Core Engine                         │
│  - Mesh handling                                │
│  - Element library (Hex, Tet)                   │
│  - Material models (Elastic, Neural)            │
│  - Explicit time integration                    │
│  - I/O (Abaqus .inp, CSV, VTU)                  │
└─────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- **C++ compiler** with C++17 support (GCC, Clang, MSVC)
- **CMake** >= 3.12
- **Python** >= 3.7
- **PyBind11** (for Python bindings)
- **Qt5** (for GUI)
- **PyVista** (for 3D visualization)

### Building from Source

```powershell
# Clone repository
cd FEM-ML

# Create build directory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Install (optional)
cmake --install build
```

### Python Package Installation

```powershell
# Install Python dependencies
pip install numpy matplotlib pyvista PyQt5 pybind11

# Build Python module
cd build
cmake --build . --target femml
```

## Usage

### 1. C++ Executable

```powershell
# Create configuration file (config.inp)
# Run solver
.\build/bin\femml_solver config.inp
```

### 2. Python API

```python
import femml

# Import mesh
importer = femml.AbaqusImporter()
mesh = importer.import_mesh("model.inp")

# Create material
material = femml.create_aluminum()

# Setup solver
solver = femml.ExplicitSolver(mesh)
solver.set_material(material)

# Add boundary conditions
bc = femml.BoundaryCondition()
bc.type = femml.BCType.FIXED
bc.nodes = [1, 2, 3, 4]
bc.component = -1
solver.add_boundary_condition(bc)

# Add load
load = femml.Load()
load.type = femml.LoadType.FORCE
load.nodes = [8]
load.component = 1
load.value = 5e5
solver.add_load(load)

# Set parameters
params = femml.SolverParams()
params.time_step = 5e-8
params.num_steps = 2000
params.auto_time_step = True
solver.set_parameters(params)

# Solve
solver.initialize()
solver.solve()

# Get results
displacements = solver.get_displacements()
solver.write_results("results.csv")
```

### 3. GUI Application

```powershell
python src/python/gui/main_window.py
```

**GUI Workflow:**
1. **File → Import Mesh**: Load Abaqus .inp file
2. **Model → Create Material**: Define materials
3. **Model → Create Load/BC**: Apply loads and boundary conditions
4. **Job → Create Job**: Setup analysis job
5. **Job → Submit Job**: Run analysis with real-time monitoring

### 4. Neural Network Integration

```python
import femml
import torch

# Create neural material
nn_material = femml.NeuralMaterial("ML-Material", rho=2700.0)

# Define inference callback
def neural_inference(total_strain):
    """Return total stress for the provided total engineering strain."""
    strain_tensor = torch.tensor(total_strain, dtype=torch.float32)
    with torch.no_grad():
        stress = model(strain_tensor).numpy()
    return stress

# Set callback (solver converts strain increments to totals before calling)
nn_material.set_inference_callback(neural_inference)

# Use in solver
solver.set_material(nn_material)
```

## Examples

### Simple Cube Under Tension

```powershell
cd examples/simple_cube
python run_example.py
```

This example demonstrates:
- Mesh import from Abaqus .inp
- Material assignment
- Boundary conditions (fixed bottom)
- Applied loads (tension on top)
- Displacement history plotting

### Output:
- `results.csv`: Final displacements
- `history.csv`: Time history data
- `displacement_history.png`: Plot

## File Formats

### Input Formats
- **Abaqus .inp**: Node coordinates, element connectivity, node/element sets
- **MFEM mesh**: Native MFEM mesh format (future)
- **Gmsh .msh**: Gmsh mesh format (future)

### Output Formats
- **CSV**: Simple comma-separated values
- **VTU**: VTK XML format for ParaView
- **HDF5**: Hierarchical data format (future)

## Configuration File Format

```ini
# Mesh
mesh model.inp

# Material
material aluminum
E 70e9
nu 0.33
rho 2700

# Time integration
dt 5e-8
steps 2000
auto_timestep true
damping 0.0

# Output
output_interval 50
output results.csv

# Boundary conditions
fixed 1 2 5 6

# Loads
force 8 1 5e5
```

## Performance

Typical performance on a modern CPU:
- **Single element**: < 1 μs per time step
- **1,000 elements**: ~10 ms per time step
- **10,000 elements**: ~100 ms per time step
- **100,000 elements**: ~1-2 s per time step

## Roadmap

### Short Term
- [ ] Complete VTU output format
- [ ] Add more material models (hyperelastic, plasticity)
- [ ] ONNX runtime integration
- [ ] Mesh refinement tools

### Medium Term
- [ ] Contact mechanics
- [ ] Shell and beam elements
- [ ] Parallel computing (OpenMP, MPI)
- [ ] GPU acceleration

### Long Term
- [ ] MFEM full integration
- [ ] Multiphysics coupling
- [ ] Topology optimization
- [ ] Cloud-based solving

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **MFEM**: Modular Finite Element Methods library
- **PyBind11**: Python/C++ bindings
- **PyVista**: 3D visualization in Python
- **Qt**: Cross-platform GUI framework

## Contact

For questions, issues, or suggestions:
- GitHub Issues: [github.com/yourname/fem-ml/issues](https://github.com/yourname/fem-ml/issues)
- Email: your.email@example.com

---

**Built with ❤️ for the FEM and ML communities**
