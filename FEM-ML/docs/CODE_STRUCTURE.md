# FEM-ML Code Structure

Complete documentation of all source files and their functionality.

## Directory Structure

```
FEM-ML/
├── include/           # C++ Headers
├── src/core/          # C++ Implementation
├── src/python/        # Python Bindings & GUI
├── examples/          # Example Problems
├── docs/              # Documentation
└── build/             # Build Output
```

---

## C++ Core Engine

### Headers (`include/`)

#### **Mesh Module**

**`mesh/Mesh.hpp`**
- **Node** struct: Stores node ID and 3D coordinates
- **ElementConnectivity** struct: Element ID, type (C3D8/C3D4), node IDs
- **NodeSet** struct: Named collection of nodes (for BCs)
- **ElementSet** struct: Named collection of elements (for materials)
- **Mesh** class: Main mesh container
  - `AddNode()` - Add node to mesh
  - `AddElement()` - Add element to mesh
  - `GetNode(id)` - Retrieve node by ID
  - `GetElement(id)` - Retrieve element by ID
  - `GetNumNodes()` - Count nodes
  - `GetNumElements()` - Count elements

**`mesh/AbaqusImporter.hpp`**
- **AbaqusImporter** class: Parses Abaqus .inp files
  - `Import(filename)` - Read mesh from file
  - `ParseNodes()` - Read *Node section
  - `ParseElements()` - Read *Element section
  - `ParseNodeSet()` - Read *Nset section
  - `ParseElementSet()` - Read *Elset section

#### **Material Module**

**`material/Material.hpp`**
- **Material** base class: Abstract interface for all materials
  - `ComputeStressIncrement(Δε, ε, σ)` - Pure virtual: Δε → Δσ
  - `ComputeTangent(strain)` - Tangent stiffness matrix
  - `GetDensity()` - Material density
  - `GetName()` - Material name

**`material/LinearElastic.hpp`**
- **LinearElastic** class: Isotropic elastic material
  - Constructor: `(name, E, nu, rho)`
  - `ComputeStressIncrement()` - Δσ = C·Δε using Lamé parameters
  - `ComputeTangent()` - Returns 6×6 stiffness matrix
  - Helper functions: `CreateAluminum()`, `CreateSteel()`, `CreateTitanium()`

**`material/NeuralMaterial.hpp`**
- **NeuralMaterial** class: ML-based material
  - `SetInferenceCallback(callback)` - Set Python/C++ inference function
  - `LoadONNXModel(path)` - Load ONNX model (future)
  - `SetFallbackMaterial(material)` - Backup if NN fails
  - `ComputeStressIncrement()` - Calls inference callback for Δσ (Abaqus-style)

#### **Element Module**

**`element/Element.hpp`**
- **Element** base class: Abstract FE element
  - `ComputeInternalForce()` - f_int = B^T·σ·V
  - `ComputeMassMatrix()` - Lumped mass
  - `ComputeVolume()` - Element volume
  - `ComputeCriticalTimeStep()` - Stability limit
  - `GetStrain()` - Current strain state
  - `GetStress()` - Current stress state

**`element/HexElement.hpp`**
- **HexElement** class: 8-node hexahedron (C3D8)
  - Single-point Gaussian integration
  - `ComputeShapeGradients()` - ∂N/∂x at center
  - `BuildBMatrix()` - 6×24 strain-displacement matrix
  - `ComputeCharacteristicLength()` - Minimum edge length

**`element/TetElement.hpp`**
- **TetElement** class: 4-node tetrahedron (C3D4)
  - Constant strain element
  - `ComputeShapeGradients()` - Constant ∂N/∂x
  - `BuildBMatrix()` - 6×12 strain-displacement matrix

#### **Solver Module**

**`solver/ExplicitSolver.hpp`**
- **ExplicitSolver** class: Explicit dynamics solver
  - `SetMaterial(material)` - Assign material to all elements
  - `AddBoundaryCondition(bc)` - Fix nodes
  - `AddLoad(load)` - Apply forces
  - `Initialize()` - Setup solver state
  - `Solve()` - Run full simulation
  - `Step()` - Single time step
  - `GetDisplacements()` - Nodal displacements
  - `GetKineticEnergy()` - System KE
  - Data structures:
    - `displacements_[n]` - 3D displacement per node
    - `velocities_[n]` - 3D velocity per node
    - `accelerations_[n]` - 3D acceleration per node
    - `masses_[n]` - Lumped mass per node

**Solver Algorithm:**
```
For each time step:
1. Compute strains from displacements
2. Compute stress increments from strain increments (material update)
3. Compute internal forces from stresses
4. Apply external forces
5. Update accelerations: a = (F_ext - F_int - damping·v) / m
6. Update velocities: v += a·dt
7. Update displacements: u += v·dt
8. Enforce boundary conditions
```

#### **Integration Module**

**`integration/CentralDifference.hpp`**
- **CentralDifference** class: Explicit time integrator
  - `Update()` - Perform one time step
  - Central difference scheme for stability

#### **I/O Module**

**`io/ResultsWriter.hpp`**
- **ResultsWriter** class: Export results
  - `WriteStep()` - Write time step data
  - `WriteField()` - Write scalar/vector fields
  - Supports: CSV, VTU (VTK XML)

- **HistoryWriter** class: Time history output
  - `SetNodes()` - Select nodes to track
  - `WriteStep()` - Write time, displacement data

---

## C++ Implementation (`src/core/`)

### **Mesh Implementation**

**`mesh/Mesh.cpp`**
- Implements Mesh class methods
- Node/element storage using `std::map<int, T>`
- Bounding box computation

**`mesh/AbaqusImporter.cpp`**
- Parses Abaqus .inp text format
- Handles multi-line element definitions
- String utilities: `Trim()`, `ToUpper()`, `Split()`

### **Material Implementation**

**`material/LinearElastic.cpp`**
- Computes Lamé parameters: λ, μ from E, ν
- Stress calculation: σ_ij = λ·δ_ij·ε_kk + 2μ·ε_ij
- Preset materials with standard properties

**`material/NeuralMaterial.cpp`**
- Callback-based inference
- Exception handling with fallback
- Future: ONNX runtime integration

### **Element Implementation**

**`element/HexElement.cpp`**
- **Shape functions** (trilinear):
  - N_i = 1/8·(1 + ξ_i·ξ)·(1 + η_i·η)·(1 + ζ_i·ζ)
- **Jacobian** computation: J = ∂N/∂ξ · coords
- **B-matrix** assembly for strain calculation
- **Internal force**: f = B^T · σ · volume

**`element/TetElement.cpp`**
- Linear tetrahedral element
- Constant strain/stress
- Analytical volume: |det(J)| / 6

### **Solver Implementation**

**`solver/ExplicitSolver.cpp`**
- Element assembly loop
- Global DOF management
- Boundary condition enforcement
- Time step stability check
- Progress callbacks

---

## Python Layer (`src/python/`)

### **API Bindings**

**`api/femml_bindings.cpp`**
- PyBind11 module definition
- Exposes C++ classes to Python:
  - `Mesh`, `Node`, `ElementConnectivity`
  - `Material`, `LinearElastic`, `NeuralMaterial`
  - `ExplicitSolver`, `SolverParams`
  - `BoundaryCondition`, `Load`
  - Enums: `BCType`, `LoadType`

### **GUI Application**

**`gui/main_window.py`** (~1000 lines)

**Classes:**

1. **MaterialDialog**
   - Material creation dialog
   - Preset selection (Al, Steel, Ti)
   - Property editing (E, ν, ρ)

2. **BCDialog**
   - Boundary condition dialog
   - Node range parsing (e.g., "1-10")
   - Component selection (X, Y, Z, All)

3. **LoadDialog**
   - Load application dialog
   - Direction and magnitude input

4. **SolverParamsDialog**
   - Solver parameter editor
   - Time step, damping, output settings

5. **ModelTree**
   - Hierarchical model view
   - Categories: Parts, Materials, Steps, Mesh, Jobs

6. **JobMonitor**
   - Real-time progress display
   - Energy monitoring
   - Progress bar

7. **Viewer3D**
   - PyVista 3D visualization
   - Mesh display with edges
   - Displacement contours
   - Deformed shape

8. **MainWindow**
   - Main application window
   - Menu system
   - Toolbar
   - Dock panels
   - Workflow orchestration

**Features:**
- Import mesh files
- Create materials/BCs/loads via dialogs
- Submit and monitor jobs
- Real-time 3D visualization
- Export results

---

## Examples (`examples/`)

### **simple_cube/**

**`cube.inp`**
- Single hexahedral element mesh
- 8 nodes forming unit cube
- Abaqus format

**`run_example.py`**
- Complete workflow example
- Imports mesh, creates material, runs solver
- Generates plots and CSV output

**`test_single_element.py`** (NEW)
- Quick test script
- Runs single element
- Creates contour plot
- Bypasses GUI

**`main.cpp`**
- C++ example
- Same workflow in C++

---

## Build System

**`CMakeLists.txt`** (root)
- Project configuration
- Options: BUILD_PYTHON, BUILD_EXAMPLES, etc.
- Find packages: pybind11, Python
- Add subdirectories

**`src/core/CMakeLists.txt`**
- Core library compilation
- Solver executable
- Link dependencies

**`src/python/CMakeLists.txt`**
- Python module compilation
- pybind11 integration

**`build.ps1`**
- PowerShell build script
- Architecture selection
- Dependency detection
- One-click build

---

## Data Flow

```
1. Import: .inp file → Mesh object
2. Setup: Mesh → Elements → Solver
3. Material: Material model → Elements
4. BC/Load: Constraints → Solver
5. Initialize: Mass matrix, time step
6. Solve: Time loop → State update
7. Output: Displacements → CSV/VTU
```

---

## Key Algorithms

### **Explicit Time Integration**
```
Initialize: u⁰, v⁰, a⁰
For n = 0 to N:
  ε^n = B·u^n
  Δε = ε^n - ε^{n-1}
  Δσ = Material(Δε, ε^{n-1}, σ^{n-1})
  σ^n = σ^{n-1} + Δσ
  f_int = ∫B^T·σ dV
  a^n = (f_ext - f_int - c·v^n) / m
  v^(n+1) = v^n + a^n·Δt
  u^(n+1) = u^n + v^(n+1)·Δt
  Enforce BCs on u, v
```

### **Stability Limit**
```
Δt_crit = L_min / c_wave
c_wave = √(E / ρ)
L_min = minimum element size
```

---

## File Formats

### **Input: Abaqus .inp**
```
*Node
1, 0.0, 0.0, 0.0
...
*Element, type=C3D8
1, 1, 2, 3, 4, 5, 6, 7, 8
...
*Nset, nset=BOTTOM
1, 2, 3, 4
```

### **Output: CSV**
```
NodeID,X,Y,Z,Ux,Uy,Uz
1,0.0,0.0,0.0,0.0,0.0,0.0
...
```

---

## Performance Notes

- **Element assembly:** O(n_elements)
- **Time step:** O(n_nodes)
- **Memory:** ~100 KB per element
- **Typical:** 1K elements ~ 10ms/step

---

For API usage, see [API_REFERENCE.md](API_REFERENCE.md)
For user guide, see [USER_GUIDE.md](USER_GUIDE.md)
