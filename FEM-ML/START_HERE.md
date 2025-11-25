# ⚡ START HERE - FEM-ML Quick Guide

## 🎯 What to Do RIGHT NOW

### **Step 1: Build** (Takes 3-5 minutes)

```powershell
# Open PowerShell
cd D:/Research-work/ML-FEM/FEM-ML

# Run build script
.\build.ps1
```

**Wait for:** `Build Complete!` message

---

### **Step 2: Run Simple Cube Example** (Takes 1 minute)

```powershell
# Go to example directory
cd examples/simple_cube

# Run it!
python run_example.py
```

**You'll see:**
```
============================================================
FEM-ML Simple Cube Example
============================================================

1. Importing mesh...
   Parsed 8 nodes
   Parsed 1 elements

2. Creating material...
   Material: Aluminum

3. Running analysis...
   Step 0 / 2000 (0.0%)
   Step 50 / 2000 (2.5%)
   ...
   Step 2000 / 2000 (100.0%)

Analysis complete!
============================================================
```

**Files created:**
- ✅ `results.csv` - Final displacements
- ✅ `history.csv` - Time history
- ✅ `displacement_history.png` - Plot

---

### **Step 3: Launch the GUI** (Takes 30 seconds)

```powershell
# From FEM-ML root directory
cd D:/Research-work/ML-FEM/FEM-ML

# Launch GUI
python src/python\gui/main_window.py
```

**GUI Opens!** Now try this workflow:

#### **GUI Workflow (5 minutes total):**

1. **Import Mesh** (1 min)
   - Click: **File → Import Mesh**
   - Navigate to: `examples/simple_cube/cube.inp`
   - Click: **Open**
   - ✅ See cube in 3D!

2. **Create Material** (30 sec)
   - Click: **Model → Create Material**
   - Select: **Preset: Aluminum**
   - Click: **OK**
   - ✅ See "Aluminum" in tree

3. **Create BC** (30 sec)
   - Click: **Model → Create Boundary Condition**
   - Nodes: `1,2,5,6` (bottom face)
   - Type: **Fixed (U1=U2=U3=0)**
   - Click: **OK**
   - ✅ See "BC-1" in tree

4. **Create Load** (30 sec)
   - Click: **Model → Create Load**
   - Nodes: `3,4,7,8` (top face)
   - Direction: **Y**
   - Magnitude: `5e5` (default)
   - Click: **OK**
   - ✅ See "Load-1" in tree

5. **Run Simulation!** (2 min)
   - Click: **Job → Submit Job** (or press **F5**)
   - ✅ Watch:
     - Progress bar fills up
     - Time updates
     - Energy displays
     - 3D mesh deforms in real-time!
   - ✅ "Analysis complete!" popup

6. **Export Results** (10 sec)
   - Click: **File → Export Results**
   - Save as: `my_results.csv`
   - Done!

---

## 📊 What the Example Does

**Simple Cube Test:**
- 1m × 1m × 1m aluminum cube
- Bottom fixed (4 nodes)
- Top pulled upward (500 kN force)
- Runs 2000 time steps
- Outputs displacement

**Expected:** Top moves ~2-3mm upward, bottom stays fixed

---

## 🔧 If Something Goes Wrong

### **Error: "femml module not found"**

**Solution:**
```powershell
# Make sure you built first!
cd D:/Research-work/ML-FEM/FEM-ML
.\build.ps1

# Then try again
cd examples/simple_cube
python run_example.py
```

---

### **Error: "No module named 'PyQt5'"**

**Solution:**
```powershell
pip install PyQt5 pyvista pyvistaqt numpy matplotlib
```

---

### **Error: "CMake not found"**

**Solution:**
- Download from: https://cmake.org/download/
- Install and add to PATH
- Run `.\build.ps1` again

---

### **GUI shows "PyVista not available"**

**Solution:**
```powershell
pip install pyvista pyvistaqt
```
Then relaunch GUI

---

## 📁 What Files Matter

### **Documentation (Read These):**
- `START_HERE.md` ← You are here!
- `HOW_TO_RUN.md` - Detailed step-by-step guide
- `README.md` - Complete reference
- `GET_STARTED.md` - Quick reference

### **Code:**
- `examples/simple_cube/` - Working example
- `src/python/gui/main_window.py` - GUI application
- `src/core/` - C++ solver code

### **Build:**
- `build.ps1` - Build script
- `CMakeLists.txt` - Build configuration

---

## ✅ Success Checklist

After running the example, you should have:

- [ ] Built successfully (saw "Build Complete!")
- [ ] Ran `run_example.py` without errors
- [ ] See `results.csv` file
- [ ] See `displacement_history.png` image
- [ ] Plot shows oscillating displacement
- [ ] GUI opens and displays welcome message
- [ ] Can import cube.inp in GUI
- [ ] See 3D cube visualization
- [ ] Can run simulation in GUI
- [ ] See real-time progress updates

**If ALL checked:** ✅ **You're ready to use FEM-ML!**

---

## 🚀 What's Next?

### **Easy Next Steps:**

1. **Modify the Example**
   - Edit `run_example.py`
   - Change material to steel
   - Increase force to 1e6 N
   - Run again and compare!

2. **Try Your Own Mesh**
   - Replace `cube.inp` with your Abaqus file
   - Import in GUI
   - Run analysis!

3. **Import Job-1**
   ```python
   mesh = femml.AbaqusImporter().import_mesh(
       "D:/Research-work/ML-FEM/Job-1 (1).inp"
   )
   ```

### **Advanced:**

4. **Add Neural Network**
   - Train ML model on FEM data
   - Create `NeuralMaterial`
   - Replace elastic material
   - Compare results!

5. **Batch Analysis**
   - Loop over materials
   - Loop over loads
   - Generate data for ML training

---

## 💡 Key Features You Have

✅ **Multi-element solver** - Handle thousands of elements
✅ **C++ performance** - Fast explicit dynamics
✅ **Python scripting** - Easy automation
✅ **Neural networks** - ML material models
✅ **Professional GUI** - Like Abaqus CAE
✅ **3D visualization** - Real-time deformation
✅ **Abaqus import** - Use existing meshes

---

## 📞 Need Help?

### **Quick Reference:**
```powershell
# Build
cd D:/Research-work/ML-FEM/FEM-ML
.\build.ps1

# Run example
cd examples/simple_cube
python run_example.py

# Launch GUI
cd D:/Research-work/ML-FEM/FEM-ML
python src/python\gui/main_window.py
```

### **Documentation:**
- **HOW_TO_RUN.md** - Full step-by-step guide
- **QUICKSTART.md** - 15-minute tutorial
- **README.md** - Complete reference

### **GUI Help:**
- **Help → How to Run** (in GUI menu)
- **Help → About** (version info)

---

## 🎉 You're Ready!

**Your explicit FEM solver with neural network support is built and ready!**

**Try it now:**
```powershell
cd D:/Research-work/ML-FEM/FEM-ML/examples/simple_cube
python run_example.py
```

**Then launch the GUI:**
```powershell
cd D:/Research-work/ML-FEM/FEM-ML
python src/python\gui/main_window.py
```

**Have fun simulating!** 🚀
