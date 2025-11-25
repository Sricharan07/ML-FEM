# FEM-ML Quick Start Guide

## Installation (10 minutes)

### 1. Prerequisites

Install the following software:

**Windows:**
```powershell
# Install Visual Studio 2019 or later with C++ support
# Install CMake: https://cmake.org/download/
# Install Python 3.7+: https://www.python.org/downloads/

# Install Python packages
pip install numpy matplotlib pyvista PyQt5 pybind11
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake python3 python3-pip qt5-default

# Python packages
pip3 install numpy matplotlib pyvista PyQt5 pybind11
```

### 2. Build FEM-ML

```powershell
# Windows
cd FEM-ML
.\build.ps1

# Linux
cd FEM-ML
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Build takes ~2-5 minutes.

## Your First Simulation (5 minutes)

### Example 1: Simple Cube Under Tension

This example simulates a 1×1×1 m aluminum cube with:
- Bottom face fixed
- Top face pulled upward with 500 kN force

```powershell
cd examples/simple_cube
python run_example.py
```

**Output:**
- `results.csv`: Final displacements at all nodes
- `history.csv`: Displacement history over time
- `displacement_history.png`: Plot of top node displacement

**Expected Result:**
- Top nodes move upward by ~2-3 mm
- Bottom nodes remain at zero
- Plot shows oscillating displacement (dynamic response)

### Example 2: Using the GUI

```powershell
python src/python/gui/main_window.py
```

**Steps:**
1. **Import Mesh**: File → Import Mesh → Select `examples/simple_cube/cube.inp`
2. **View Mesh**: Rotate view with mouse, zoom with scroll wheel
3. **Create Material**: Model → Create Material → Select Aluminum
4. **Run Analysis**: Job → Submit Job
5. **Watch Progress**: See real-time displacement visualization

## Understanding the Results

### Displacement History

The `displacement_history.png` plot shows:
- **X-axis**: Time in microseconds (μs)
- **Y-axis**: Displacement in millimeters (mm)
- **Oscillations**: Natural vibration of the cube (explicit dynamics)

Why oscillations?
- Explicit dynamics captures wave propagation
- No damping applied → free vibration
- To get steady-state: add damping or run longer

### Results CSV

Open `results.csv` in Excel or Python:

```python
import pandas as pd
df = pd.read_csv('results.csv')
print(df)
```

Columns:
- `NodeID`: Node number
- `X, Y, Z`: Original coordinates
- `Ux, Uy, Uz`: Displacement in X, Y, Z directions

## Customizing the Analysis

### Change Material Properties

Edit `config.inp`:
```ini
# Steel instead of aluminum
E 210e9
nu 0.3
rho 7800
```

### Increase Mesh Density

Create finer mesh in Abaqus or Gmsh, export as `.inp`:
```
*Node
1, 0, 0, 0
2, 0.5, 0, 0
3, 1.0, 0, 0
...
*Element, type=C3D8
1, 1, 2, 10, 9, 5, 6, 14, 13
...
```

### Add Neural Network Material

```python
import femml
import torch

# Load your trained model
model = torch.load('my_material_model.pt')

# Create neural material
nn_mat = femml.NeuralMaterial("ML-Material", rho=2700)

def inference(total_strain):
    strain_t = torch.tensor(total_strain, dtype=torch.float32)
    with torch.no_grad():
        stress = model(strain_t).numpy()
    return stress

nn_mat.set_inference_callback(inference)
solver.set_material(nn_mat)
```

## Next Steps

### Learn More
- Read [README.md](README.md) for complete documentation
- Study [CODE_STRUCTURE.md](CODE_STRUCTURE.md) for design details
- Explore examples in `examples/` directory

### Run Your Own Models
1. **Create mesh** in Abaqus/Gmsh
2. **Export as .inp** file
3. **Run with FEM-ML**:
   ```powershell
   .\build/bin\femml_solver.exe my_config.inp
   ```

### Develop Extensions
- Add new material models in `src/core/material/`
- Add new element types in `src/core/element/`
- Contribute via GitHub pull requests

## Troubleshooting

### Build Errors

**"CMake not found"**
- Install CMake and add to PATH

**"pybind11 not found"**
```powershell
pip install pybind11
```

**"Qt not found"**
```powershell
# Windows: Install Qt from https://www.qt.io/download
# Linux: sudo apt-get install qt5-default
```

### Runtime Errors

**"femml module not found"**
- Make sure you're running Python from FEM-ML directory
- Check that `build/femml.pyd` (Windows) or `build/femml.so` (Linux) exists

**"Mesh import failed"**
- Check .inp file format is correct
- Ensure node IDs are sequential starting from 1

**"Simulation diverges"**
- Time step too large: enable `auto_timestep true`
- Add damping: `damping 0.1`
- Check mesh quality (no inverted elements)

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: See `docs/` folder
- **Examples**: Study working examples in `examples/`

## Performance Tips

- **Enable auto time step**: Prevents instability
- **Use OpenMP** (future): Parallel element assembly
- **Reduce output**: Increase `output_interval` for faster runs
- **GPU acceleration** (future): For neural network inference

---

**Welcome to FEM-ML!** 🚀

For more information, see [README.md](README.md)
