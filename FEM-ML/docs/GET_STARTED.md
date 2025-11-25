# 🚀 GET STARTED WITH FEM-ML

## What You Have Now

### ✅ **Complete Explicit FEM Solver**
- Multi-element support (hex & tet)
- Stress, strain, displacement computation
- Neural network integration
- Abaqus-like GUI
- Python & C++ APIs

---

## 3 Ways to Use FEM-ML

### 1️⃣ **GUI Application** (Easiest)
```powershell
python src/python\gui/main_window.py
```
- Import mesh (File → Import)
- Create materials
- Set loads & BCs
- Run simulation
- View results in 3D

**Perfect for:** Interactive modeling, visualization

---

### 2️⃣ **Python API** (Most Flexible)
```python
import femml

# Import mesh
mesh = femml.AbaqusImporter().import_mesh("model.inp")

# Create material
material = femml.create_aluminum()

# Setup solver
solver = femml.ExplicitSolver(mesh)
solver.set_material(material)

# Add BC (fix nodes 1,2,3,4)
bc = femml.BoundaryCondition()
bc.type = femml.BCType.FIXED
bc.nodes = [1, 2, 3, 4]
solver.add_boundary_condition(bc)

# Add load (500kN on node 8, Y direction)
load = femml.Load()
load.type = femml.LoadType.FORCE
load.nodes = [8]
load.component = 1
load.value = 5e5
solver.add_load(load)

# Run
params = femml.SolverParams()
params.num_steps = 2000
params.auto_time_step = True
solver.set_parameters(params)
solver.initialize()
solver.solve()

# Get results
displacements = solver.get_displacements()
solver.write_results("results.csv")
```

**Perfect for:** Scripting, automation, batch runs

---

### 3️⃣ **C++ Executable** (Fastest)
```powershell
# Create config.inp
# Run solver
.\build/bin\femml_solver.exe config.inp
```

**Perfect for:** Production runs, performance

---

## 🎯 Quick Start (15 Minutes)

### Step 1: Build (5 min)
```powershell
cd D:/Research-work/ML-FEM/FEM-ML
.\build.ps1
```

### Step 2: Run Example (5 min)
```powershell
cd examples/simple_cube
python run_example.py
```

### Step 3: View Results (5 min)
- Open `displacement_history.png`
- Open `results.csv` in Excel
- Check console output

---

## 📁 Project Structure

```
FEM-ML/
├── 📖 QUICKSTART.md           ← Read this first!
├── 📖 README.md               ← Full documentation
├── 📖 CODE_STRUCTURE.md         ← System design
├── 📖 PROJECT_SUMMARY.md      ← What was built
│
├── 🔨 build.ps1               ← Build script
├── ⚙️ CMakeLists.txt          ← Build config
│
├── 📂 include/                ← C++ headers
├── 📂 src/core/               ← C++ implementation
├── 📂 src/python/             ← Python API & GUI
│   ├── api/                   ← PyBind11 bindings
│   └── gui/                   ← Qt application
│
├── 📂 examples/               ← Example problems
│   └── simple_cube/           ← Start here!
│       ├── cube.inp
│       ├── config.inp
│       ├── run_example.py     ← Python example
│       └── main.cpp           ← C++ example
│
└── 📂 build/                  ← Build output
    ├── bin/                   ← Executables
    └── femml.pyd              ← Python module
```

---

## 🎓 Learning Path

### Beginner (Day 1)
1. ✅ Build the project
2. ✅ Run simple_cube example
3. ✅ Understand results
4. ✅ Launch GUI

### Intermediate (Week 1)
1. Import your own mesh
2. Modify materials
3. Change loads/BCs
4. Run parameter studies

### Advanced (Month 1)
1. Integrate neural networks
2. Add custom materials
3. Implement new elements
4. Contribute to codebase

---

## 🧪 Test Your Installation

### Quick Test
```powershell
# Run C++ example
cd examples/simple_cube
..\..build/bin\example_cube.exe

# Run Python example
python run_example.py

# Launch GUI
python ..\..\src/python\gui/main_window.py
```

**Expected Output:**
- ✅ Console shows solver progress
- ✅ Results files created
- ✅ Plot generated
- ✅ GUI opens

---

## 🔧 Your Research Workflow

### 1. Generate Training Data
```python
# Run FEM with various materials
for E in [50e9, 70e9, 100e9]:
    material = femml.LinearElastic("Mat", E, 0.33, 2700)
    solver.set_material(material)
    solver.solve()
    # Collect strain/stress pairs
    data.append((strain, stress))
```

### 2. Train Neural Network
```python
import torch
# Train model on FEM data
model = MyNeuralNetwork()
model.train(data)
torch.save(model, "material_model.pt")
```

### 3. Use in FEM
```python
# Load trained model
model = torch.load("material_model.pt")

# Create neural material
nn_mat = femml.NeuralMaterial("ML-Material", rho=2700)
nn_mat.set_inference_callback(lambda total_strain: model(total_strain))  # returns total stress values

# Run FEM with ML material
solver.set_material(nn_mat)
solver.solve()
```

### 4. Compare & Publish
```python
# Compare classical vs ML
classical_results = run_with_elastic()
ml_results = run_with_neural()
plot_comparison(classical_results, ml_results)
```

---

## 📊 What You Can Do Now

### ✅ Mesh & Modeling
- [x] Import Abaqus .inp meshes
- [x] Multi-element simulations
- [x] Node/element sets
- [x] Surfaces

### ✅ Materials
- [x] Linear elastic (Al, Steel, Ti)
- [x] Custom elastic materials
- [x] Neural network materials
- [x] Hybrid physics-ML

### ✅ Analysis
- [x] Explicit dynamics
- [x] Automatic time stepping
- [x] Energy monitoring
- [x] Displacement output

### ✅ Visualization
- [x] 3D mesh display
- [x] Displacement contours
- [x] Time history plots
- [x] Real-time updates

---

## 🎯 Your Job-1 Analysis

```python
# Import your actual mesh
mesh = femml.AbaqusImporter().import_mesh(
    "D:/Research-work/ML-FEM/Job-1 (1).inp"
)

print(f"Nodes: {mesh.get_num_nodes()}")
print(f"Elements: {mesh.get_num_elements()}")

# Setup analysis
solver = femml.ExplicitSolver(mesh)
material = femml.create_aluminum()
solver.set_material(material)

# ... add your BCs and loads

# Run
solver.initialize()
solver.solve()

# Results
solver.write_results("job1_results.csv")
```

---

## 💡 Tips & Tricks

### Performance
- Enable `auto_time_step = true` for stability
- Increase `output_interval` for faster runs
- Use C++ executable for production

### Accuracy
- Check mesh quality (no inverted elements)
- Validate with simple test cases
- Compare with analytical solutions

### Neural Networks
- Train on diverse loading conditions
- Use fallback material for safety
- Validate NN predictions offline first

---

## 🆘 Getting Help

### Documentation
1. **QUICKSTART.md** - Tutorial
2. **README.md** - Reference
3. **CODE_STRUCTURE.md** - Design
4. **Examples/** - Working code

### Troubleshooting
- Build fails? Check CMake, Python versions
- Import fails? Check .inp format
- Solver crashes? Enable auto_timestep

---

## 🎉 You're Ready!

**Your complete FEM-ML system includes:**

✅ Multi-element explicit solver
✅ Neural network support
✅ Abaqus-like GUI
✅ Python & C++ APIs
✅ Full documentation
✅ Working examples

**Now go build something amazing!** 🚀

---

## Quick Commands Reference

```powershell
# Build
.\build.ps1

# Run C++ solver
.\build/bin\femml_solver.exe config.inp

# Run Python example
python examples/simple_cube/run_example.py

# Launch GUI
python src/python\gui/main_window.py

# Run with your mesh
python -c "
import femml
mesh = femml.AbaqusImporter().import_mesh('your_mesh.inp')
print(f'Loaded {mesh.get_num_nodes()} nodes')
"
```

---

**Start with:** [QUICKSTART.md](QUICKSTART.md)
**Read next:** [README.md](README.md)
**Deep dive:** [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
**Summary:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**Happy simulating! 🎊**
