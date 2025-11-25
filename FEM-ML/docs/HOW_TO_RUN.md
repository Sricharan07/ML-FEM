# 🚀 HOW TO RUN FEM-ML - STEP BY STEP

## ⚡ QUICK START (Copy & Paste Commands)

### **Step 1: Build the Project**
```powershell
cd D:/Research-work/ML-FEM/FEM-ML
.\build.ps1
```
**Wait for:** "Build Complete!" message (~2-5 minutes)

---

### **Step 2: Run the Simple Cube Example**

#### **Option A: Python Example (Recommended)**
```powershell
cd D:/Research-work/ML-FEM/FEM-ML/examples/simple_cube
python run_example.py
```

**You will see:**
```
============================================================
FEM-ML Simple Cube Example
============================================================

1. Importing mesh...
  Parsed 8 nodes
  Parsed 1 elements (type: C3D8)

2. Creating material...
   Material: Aluminum
   Density: 2700.0 kg/m³

3. Setting up solver...

4. Initializing solver...
Number of nodes: 8
Number of elements: 1
Critical time step: 1.234e-07 s
Using time step: 1.111e-07 s

5. Running analysis...
Step 0 / 2000 (0.0%) Time: 0.000000e+00 s
  Kinetic Energy: 0.000000e+00 J
Step 50 / 2000 (2.5%) Time: 5.556e-06 s
  Kinetic Energy: 1.234e-03 J
...

6. Writing results...
   Results written to results.csv

7. Plotting results...
   Saved plot: displacement_history.png

8. Final displacements:
   Node 1: (0.0000, 0.0000, 0.0000) mm
   Node 2: (0.0000, 0.0000, 0.0000) mm
   ...
   Node 8: (0.1234, 2.4567, 0.0123) mm

============================================================
Analysis complete!
============================================================
```

**Output Files Created:**
- `results.csv` - Final nodal displacements
- `history.csv` - Time history data
- `displacement_history.png` - Displacement plot

---

#### **Option B: C++ Example**
```powershell
cd D:/Research-work/ML-FEM/FEM-ML/examples/simple_cube
..\..\build/bin\Release\example_cube.exe
```

---

### **Step 3: Launch the GUI**

```powershell
cd D:/Research-work/ML-FEM/FEM-ML
python src/python\gui/main_window.py
```

**GUI Will Open:**
- 3D viewer in center
- Model tree on left
- Properties panel on right
- Job monitor at bottom

**Try This in GUI:**
1. Click **File → Import Mesh**
2. Navigate to: `examples/simple_cube/cube.inp`
3. Click **Open**
4. See the cube appear in 3D viewer!
5. Click **Job → Submit Job** to run simulation
6. Watch real-time results!

---

## 📋 DETAILED INSTRUCTIONS

### **Understanding the Simple Cube Example**

#### **What It Does:**
- Simulates a **1m × 1m × 1m aluminum cube**
- **Bottom face** (nodes 1,2,5,6) is **fixed**
- **Top face** (nodes 3,4,7,8) is **pulled upward** with **500 kN force**
- Runs **2000 time steps** of explicit dynamics
- Outputs displacement at each node

#### **Expected Result:**
- **Top nodes move ~2-3 mm upward**
- **Bottom nodes stay at zero**
- **Oscillations** due to dynamic effects (wave propagation)

---

### **Files Explained**

#### `cube.inp` - Mesh File
```
*Node
1, 0.0, 0.0, 0.0   ← Bottom-left-front
2, 1.0, 0.0, 0.0   ← Bottom-right-front
3, 1.0, 1.0, 0.0   ← Bottom-right-back
4, 0.0, 1.0, 0.0   ← Bottom-left-back
5, 0.0, 0.0, 1.0   ← Top-left-front
6, 1.0, 0.0, 1.0   ← Top-right-front
7, 1.0, 1.0, 1.0   ← Top-right-back
8, 0.0, 1.0, 1.0   ← Top-left-back

*Element, type=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
```

#### `run_example.py` - Python Script
```python
# Imports mesh
mesh = femml.AbaqusImporter().import_mesh("cube.inp")

# Creates aluminum material
material = femml.create_aluminum()

# Sets up solver
solver = femml.ExplicitSolver(mesh)
solver.set_material(material)

# Fixes bottom (nodes 1,2,5,6)
bc.nodes = [1, 2, 5, 6]
bc.type = femml.BCType.FIXED

# Pulls top (nodes 3,4,7,8) upward
for node in [3, 4, 7, 8]:
    load.nodes = [node]
    load.component = 1  # Y direction
    load.value = 5e5     # 500 kN

# Runs simulation
solver.solve()
```

---

## 🔧 TROUBLESHOOTING

### **Problem: "femml module not found"**

**Solution:**
```powershell
# Make sure you're in the example directory
cd D:/Research-work/ML-FEM/FEM-ML/examples/simple_cube

# The script automatically finds the built module
python run_example.py
```

**Why:** The `run_example.py` script automatically adds build paths to sys.path

---

### **Problem: "Build failed"**

**Check CMake:**
```powershell
cmake --version
```
Should show version 3.12 or higher

**Check Python:**
```powershell
python --version
```
Should show 3.7 or higher

**Install missing Python packages:**
```powershell
pip install numpy matplotlib pyvista PyQt5 pybind11
```

---

### **Problem: GUI doesn't open**

**Install PyVista:**
```powershell
pip install pyvista pyvistaqt
```

**Check Qt:**
```powershell
python -c "from PyQt5.QtWidgets import QApplication; print('Qt OK')"
```

---

### **Problem: No plot generated**

**Install Matplotlib:**
```powershell
pip install matplotlib
```

---

## 📊 VIEWING RESULTS

### **1. Displacement Plot**

Open `displacement_history.png`:
- **X-axis**: Time (microseconds)
- **Y-axis**: Displacement (millimeters)
- **Shows**: Node 8 moving up and oscillating

---

### **2. Results CSV**

Open `results.csv` in Excel or Notepad:
```csv
NodeID,X,Y,Z,Ux,Uy,Uz
1,0.0,0.0,0.0,0.0,0.0,0.0
2,1.0,0.0,0.0,0.0,0.0,0.0
...
8,0.0,1.0,1.0,0.00012,0.00247,0.00001
```

**Columns:**
- `X,Y,Z` = Original position (meters)
- `Ux,Uy,Uz` = Displacement (meters)

---

### **3. History CSV**

Open `history.csv`:
```csv
Time,Ux8,Uy8,Uz8
0.0,0.0,0.0,0.0
5.556e-06,1.23e-06,4.56e-05,2.34e-07
...
```

**Use for:** Time-series plotting, further analysis

---

## 🎯 NEXT STEPS

### **Modify the Example**

#### **Change Material to Steel:**
```python
# In run_example.py, replace:
material = femml.create_aluminum()
# With:
material = femml.create_steel()
```

#### **Increase Force:**
```python
# In run_example.py, change:
load.value = 5e5  # 500 kN
# To:
load.value = 1e6  # 1000 kN (1 MN)
```

#### **Run Longer:**
```python
# Change:
params.num_steps = 2000
# To:
params.num_steps = 5000
```

---

### **Import Your Own Mesh**

```python
# Replace cube.inp with your mesh
mesh = femml.AbaqusImporter().import_mesh("D:/path/to/your_mesh.inp")
```

**Requirements:**
- Abaqus `.inp` format
- Contains `*Node` and `*Element` sections
- Supported elements: C3D8, C3D8R, C3D4

---

### **Use Your Job-1 Mesh**

```python
# Import the large mesh
mesh = femml.AbaqusImporter().import_mesh(
    "D:/Research-work/ML-FEM/Job-1 (1).inp"
)

print(f"Loaded: {mesh.get_num_nodes()} nodes")
print(f"Elements: {mesh.get_num_elements()}")

# Run analysis on full mesh
solver = femml.ExplicitSolver(mesh)
# ... continue setup ...
```

---

## 🎨 USING THE GUI

### **Complete GUI Workflow:**

#### **1. Launch GUI**
```powershell
python src/python\gui/main_window.py
```

#### **2. Import Mesh**
- Click **File → Import Mesh**
- Select `examples/simple_cube/cube.inp`
- Click **Open**
- ✅ Cube appears in 3D viewer

#### **3. Inspect Model**
- **Left Panel**: See model tree
  - Nodes: 8
  - Elements: 1
- **3D Viewer**: Rotate with mouse, zoom with scroll

#### **4. Create Material**
- Click **Model → Create Material**
- ✅ "Aluminum" appears in Materials tree

#### **5. Create Job**
- Click **Job → Create Job**
- ✅ "Job-1" appears in Jobs tree

#### **6. Submit Job**
- Click **Job → Submit Job**
- ✅ Watch real-time simulation!
  - Progress bar updates
  - Energy displayed
  - 3D mesh deforms
  - Bottom panel shows status

#### **7. View Results**
- Check **Results** files in directory
- See deformed mesh in 3D viewer

---

## ⚙️ ADVANCED USAGE

### **Neural Network Material**

```python
import torch
import femml

# Load your trained model
model = torch.load("my_material_model.pt")

# Create neural material
nn_mat = femml.NeuralMaterial("ML-Material", rho=2700)

# Define inference function
def predict_stress(total_strain):
    strain_tensor = torch.tensor(total_strain, dtype=torch.float32)
    with torch.no_grad():
        stress = model(strain_tensor).numpy()
    return stress

# Set callback
# FEM-ML passes updated total strain and expects total stress
nn_mat.set_inference_callback(predict_stress)

# Use in solver
solver.set_material(nn_mat)
solver.solve()
```

---

### **Batch Analysis**

```python
# Run multiple cases
materials = {
    "aluminum": femml.create_aluminum(),
    "steel": femml.create_steel(),
    "titanium": femml.create_titanium()
}

for name, mat in materials.items():
    solver.set_material(mat)
    solver.solve()
    solver.write_results(f"results_{name}.csv")
```

---

## 📖 MORE DOCUMENTATION

- **README.md** - Full reference
- **QUICKSTART.md** - Tutorial
- **CODE_STRUCTURE.md** - System design
- **PROJECT_SUMMARY.md** - What was built
- **GET_STARTED.md** - Quick reference

---

## ✅ CHECKLIST

Before running:
- [ ] Built project with `.\build.ps1`
- [ ] Installed Python packages: `pip install numpy matplotlib pyvista PyQt5`
- [ ] In correct directory: `examples/simple_cube`

Running example:
- [ ] Execute: `python run_example.py`
- [ ] See console output
- [ ] Check `results.csv` created
- [ ] Check `displacement_history.png` created

Using GUI:
- [ ] Execute: `python src/python\gui/main_window.py`
- [ ] Import mesh file
- [ ] See 3D visualization
- [ ] Submit job
- [ ] View results

---

## 🎉 SUCCESS CRITERIA

✅ **You're successful if you see:**

1. **Console Output:**
   ```
   Analysis complete!
   ============================================================
   ```

2. **Files Created:**
   - `results.csv` (exists, >0 bytes)
   - `history.csv` (exists, >0 bytes)
   - `displacement_history.png` (can open image)

3. **Results Make Sense:**
   - Top nodes displaced ~2-3 mm
   - Bottom nodes = 0 mm
   - Plot shows oscillations

4. **GUI Opens:**
   - Window appears
   - 3D viewer shows mesh
   - Can rotate/zoom
   - Can run simulation

---

**READY? Start with Step 1!** ⬆️
