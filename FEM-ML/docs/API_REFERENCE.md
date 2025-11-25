# FEM-ML API Reference

Complete Python API documentation for FEM-ML.

## Module: `femml`

Import the module:
```python
import femml
```

---

## Classes

### **Mesh**

Container for finite element mesh data.

#### Methods

```python
mesh = femml.Mesh()
```

**`add_node(node: Node)`**
- Add node to mesh

**`add_element(element: ElementConnectivity)`**
- Add element to mesh

**`get_node(id: int) -> Node`**
- Get node by ID
- Raises exception if not found

**`get_element(id: int) -> ElementConnectivity`**
- Get element by ID
- Raises exception if not found

**`get_num_nodes() -> int`**
- Return number of nodes

**`get_num_elements() -> int`**
- Return number of elements

**`print_summary()`**
- Print mesh statistics

---

### **AbaqusImporter**

Import mesh from Abaqus .inp files.

#### Methods

```python
importer = femml.AbaqusImporter()
mesh = importer.import_mesh("model.inp")
```

**`import_mesh(filename: str) -> Mesh`**
- Parse Abaqus .inp file
- Returns Mesh object
- Raises exception on error

---

### **Material** (Base Class)

Abstract base class for all materials.

#### Methods

**`compute_stress_increment(strain_increment: array[6], current_strain: array[6], current_stress: array[6]) -> array[6]`**
- Return stress increment for a strain increment (Abaqus-style explicit)
- All arrays use engineering Voigt order `[εxx, εyy, εzz, γxy, γxz, γyz]`
- Solver accumulates returned Δσ into the stored stress state

**`get_name() -> str`**
- Return material name

**`get_density() -> float`**
- Return density (kg/m³)

**`get_type() -> str`**
- Return material type string

---

### **LinearElastic**

Linear elastic isotropic material.

#### Constructor

```python
material = femml.LinearElastic(name, E, nu, rho)
```

**Parameters:**
- `name` (str): Material name
- `E` (float): Young's modulus (Pa)
- `nu` (float): Poisson's ratio
- `rho` (float): Density (kg/m³)

#### Methods

**`get_youngs_modulus() -> float`**
- Return E

**`get_poissons_ratio() -> float`**
- Return ν

#### Factory Functions

```python
aluminum = femml.create_aluminum()  # E=70 GPa, ν=0.33, ρ=2700
steel = femml.create_steel()        # E=210 GPa, ν=0.3, ρ=7800
titanium = femml.create_titanium()  # E=110 GPa, ν=0.34, ρ=4500
```

---

### **NeuralMaterial**

Neural network-based material model.

#### Constructor

```python
nn_mat = femml.NeuralMaterial(name, rho)
```

**Parameters:**
- `name` (str): Material name
- `rho` (float): Density (kg/m³)

#### Methods

**`set_inference_callback(callback: Callable)`**
- Set inference function
- callback: `(strain: array[6]) -> array[6]`, where `strain` is the updated total strain

**`set_fallback_material(material: Material)`**
- Set backup material if NN fails

**`load_onnx_model(path: str)`**
- Load ONNX model (future feature)

#### Example

```python
import torch

model = torch.load("model.pt")

nn_mat = femml.NeuralMaterial("ML-Material", rho=2700)

def predict(total_strain):
    with torch.no_grad():
        return model(torch.tensor(total_strain)).numpy()

nn_mat.set_inference_callback(predict)  # receives total strain, return total stress
```

---

### **ExplicitSolver**

Explicit dynamics finite element solver.

#### Constructor

```python
solver = femml.ExplicitSolver(mesh)
```

**Parameters:**
- `mesh` (Mesh): Finite element mesh

#### Methods

**`set_material(material: Material)`**
- Assign material to all elements

**`add_boundary_condition(bc: BoundaryCondition)`**
- Add boundary condition

**`add_load(load: Load)`**
- Add applied load

**`set_parameters(params: SolverParams)`**
- Set solver parameters

**`initialize()`**
- Initialize solver (call before solve)

**`solve()`**
- Run complete simulation

**`step()`**
- Run single time step

**`is_finished() -> bool`**
- Check if simulation complete

**`get_current_step() -> int`**
- Get current step number

**`get_current_time() -> float`**
- Get current simulation time (s)

**`get_displacements() -> List[array[3]]`**
- Get nodal displacements
- Returns list of [ux, uy, uz] for each node

**`get_velocities() -> List[array[3]]`**
- Get nodal velocities

**`get_node_displacement(node_id: int) -> array[3]`**
- Get displacement of specific node

**`get_node_velocity(node_id: int) -> array[3]`**
- Get velocity of specific node

**`get_kinetic_energy() -> float`**
- Get total kinetic energy (J)

**`get_strain_energy() -> float`**
- Get total strain energy (J)

**`write_results(filename: str)`**
- Export results to CSV

**`set_output_callback(callback: Callable)`**
- Set progress callback
- callback: `(step: int, time: float) -> None`

---

### **SolverParams**

Solver parameter container.

#### Constructor

```python
params = femml.SolverParams()
```

#### Attributes

**`time_step` (float)**
- Time step size (s)
- Default: 1e-7

**`num_steps` (int)**
- Number of time steps
- Default: 1000

**`damping` (float)**
- Velocity damping coefficient
- Default: 0.0

**`output_interval` (int)**
- Output every N steps
- Default: 10

**`auto_time_step` (bool)**
- Automatically compute time step
- Default: False

**`time_step_scale` (float)**
- Safety factor for auto time step
- Default: 0.9

#### Example

```python
params = femml.SolverParams()
params.time_step = 5e-8
params.num_steps = 2000
params.damping = 0.05
params.auto_time_step = True
solver.set_parameters(params)
```

---

### **BoundaryCondition**

Boundary condition specification.

#### Constructor

```python
bc = femml.BoundaryCondition()
```

#### Attributes

**`type` (BCType)**
- Boundary condition type
- Enum: `BCType.FIXED`, `BCType.DISPLACEMENT`

**`nodes` (List[int])**
- Node IDs to constrain

**`component` (int)**
- DOF component: 0=X, 1=Y, 2=Z, -1=All

**`value` (float)**
- Prescribed value (for displacement BC)

#### Example

```python
bc = femml.BoundaryCondition()
bc.type = femml.BCType.FIXED
bc.nodes = [1, 2, 3, 4]
bc.component = -1  # Fix all directions
solver.add_boundary_condition(bc)
```

---

### **Load**

Applied load specification.

#### Constructor

```python
load = femml.Load()
```

#### Attributes

**`type` (LoadType)**
- Load type
- Enum: `LoadType.FORCE`, `LoadType.PRESSURE`

**`nodes` (List[int])**
- Node IDs for nodal loads

**`component` (int)**
- Direction: 0=X, 1=Y, 2=Z

**`value` (float)**
- Load magnitude (N for force)

#### Example

```python
load = femml.Load()
load.type = femml.LoadType.FORCE
load.nodes = [8]
load.component = 1  # Y direction
load.value = 5e5    # 500 kN
solver.add_load(load)
```

---

### **Node**

Node data structure.

#### Attributes

**`id` (int)**
- Node ID

**`coords` (array[3])**
- [x, y, z] coordinates

---

### **ElementConnectivity**

Element connectivity data.

#### Attributes

**`id` (int)**
- Element ID

**`type` (str)**
- Element type: "C3D8", "C3D8R", "C3D4"

**`nodes` (List[int])**
- Node IDs in connectivity order

---

## Enums

### **BCType**

- `BCType.FIXED` - Fixed displacement (zero)
- `BCType.DISPLACEMENT` - Prescribed displacement
- `BCType.VELOCITY` - Prescribed velocity
- `BCType.ACCELERATION` - Prescribed acceleration

### **LoadType**

- `LoadType.FORCE` - Nodal force
- `LoadType.PRESSURE` - Surface pressure
- `LoadType.BODY_FORCE` - Body force
- `LoadType.GRAVITY` - Gravity load

---

## Complete Example

```python
import femml
import numpy as np

# 1. Import mesh
importer = femml.AbaqusImporter()
mesh = importer.import_mesh("model.inp")

# 2. Create material
material = femml.create_aluminum()

# 3. Create solver
solver = femml.ExplicitSolver(mesh)
solver.set_material(material)

# 4. Add boundary condition
bc = femml.BoundaryCondition()
bc.type = femml.BCType.FIXED
bc.nodes = [1, 2, 3, 4]
bc.component = -1
solver.add_boundary_condition(bc)

# 5. Add load
load = femml.Load()
load.type = femml.LoadType.FORCE
load.nodes = [8]
load.component = 1
load.value = 5e5
solver.add_load(load)

# 6. Configure
params = femml.SolverParams()
params.num_steps = 2000
params.auto_time_step = True
solver.set_parameters(params)

# 7. Solve
solver.initialize()
solver.solve()

# 8. Get results
displacements = solver.get_displacements()
disp_array = np.array([[d[0], d[1], d[2]] for d in displacements])

# 9. Export
solver.write_results("results.csv")
```

---

For usage examples, see [USER_GUIDE.md](USER_GUIDE.md)
For code details, see [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
