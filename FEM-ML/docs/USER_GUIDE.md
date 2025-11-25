# FEM-ML User Guide

Complete guide to using FEM-ML for explicit finite element analysis.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Python API](#python-api)
4. [GUI Usage](#gui-usage)
5. [Examples](#examples)
6. [Advanced Topics](#advanced-topics)

---

## Installation

### Build from Source
```powershell
cd FEM-ML
.\build.ps1
```

### Install Python Dependencies
```powershell
pip install numpy matplotlib pyvista PyQt5 pybind11
```

---

## Quick Start

### Test Single Element
```powershell
cd examples/simple_cube
python test_single_element.py
```

Creates:
- `single_element_results.png` - Contour plot
- Console output with displacements

### Run Complete Example
```powershell
cd examples/simple_cube
python run_example.py
```

---

## Python API

### Import Mesh
```python
import femml

importer = femml.AbaqusImporter()
mesh = importer.import_mesh("model.inp")

print(f"Nodes: {mesh.get_num_nodes()}")
print(f"Elements: {mesh.get_num_elements()}")
```

### Create Materials
```python
# Preset materials
aluminum = femml.create_aluminum()
steel = femml.create_steel()
titanium = femml.create_titanium()

# Custom material
custom = femml.LinearElastic("Custom", E=100e9, nu=0.3, rho=3000)
```

### Setup Solver
```python
solver = femml.ExplicitSolver(mesh)
solver.set_material(aluminum)

# Boundary condition
bc = femml.BoundaryCondition()
bc.type = femml.BCType.FIXED
bc.nodes = [1, 2, 3, 4]
bc.component = -1  # All directions
solver.add_boundary_condition(bc)

# Load
load = femml.Load()
load.type = femml.LoadType.FORCE
load.nodes = [8]
load.component = 1  # Y direction
load.value = 5e5
solver.add_load(load)
```

### Run Simulation
```python
# Set parameters
params = femml.SolverParams()
params.time_step = 5e-8
params.num_steps = 2000
params.auto_time_step = True
solver.set_parameters(params)

# Initialize and solve
solver.initialize()
solver.solve()

# Get results
displacements = solver.get_displacements()
solver.write_results("results.csv")
```

---

## GUI Usage

### Launch
```powershell
python src/python\gui/main_window.py
```

### Workflow
1. **File → Import Mesh** - Load .inp file
2. **Model → Create Material** - Select preset or custom
3. **Model → Create Boundary Condition** - Fix nodes
4. **Model → Create Load** - Apply forces
5. **Job → Submit Job** - Run simulation
6. **File → Export Results** - Save data

---

## Examples

### Example 1: Simple Cube
Location: `examples/simple_cube/`

Tests single hexahedral element under tension.

### Example 2: Your Own Mesh
```python
mesh = femml.AbaqusImporter().import_mesh("your_mesh.inp")
# Continue with solver setup...
```

---

## Advanced Topics

### Neural Network Materials
```python
import torch

# Your trained model
model = torch.load("material_model.pt")

# Create NN material
nn_mat = femml.NeuralMaterial("ML-Material", rho=2700)

# Inference callback
def predict(total_strain):
    with torch.no_grad():
        return model(torch.tensor(total_strain)).numpy()

nn_mat.set_inference_callback(predict)  # receives total strain, return total stress
solver.set_material(nn_mat)
```

### Batch Analysis
```python
materials = [
    femml.create_aluminum(),
    femml.create_steel(),
    femml.create_titanium()
]

for i, mat in enumerate(materials):
    solver.set_material(mat)
    solver.solve()
    solver.write_results(f"results_{i}.csv")
```

---

## Troubleshooting

### "femml module not found"
- Build the project first: `.\build.ps1`
- Check Python path includes build directory

### "Import failed"
- Verify .inp file format
- Check node/element numbering

### GUI doesn't open
- Install PyQt5: `pip install PyQt5`
- Install PyVista: `pip install pyvista pyvistaqt`

---

For more details, see:
- [API Reference](API_REFERENCE.md)
- [Code Structure](CODE_STRUCTURE.md)
- [Architecture](CODE_STRUCTURE.md)
