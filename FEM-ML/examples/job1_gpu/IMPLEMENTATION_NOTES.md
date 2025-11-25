# Job-1 GPU Implementation Notes

## What Was Created

A complete GPU-accelerated explicit FEM solver for the Job-1 mesh with the following components:

### 1. GPU Solver Core (`GPUExplicitSolver`)

**Location**:
- Header: `include/solver/GPUExplicitSolver.hpp`
- Implementation: `src/core/solver/GPUExplicitSolver.cu`

**Features**:
- CUDA-accelerated time integration
- GPU memory management
- Parallel element assembly
- Efficient boundary condition enforcement
- Performance metrics tracking

**Key Methods**:
```cpp
void Initialize()              // Allocate GPU memory, setup solver
void Step()                     // Single GPU-accelerated time step
void Solve()                    // Complete simulation loop
void SetDeviceID(int id)       // Select GPU device
size_t GetGPUMemoryUsage()     // Monitor memory usage
double GetGPUComputeTime()     // Get GPU computation time
```

### 2. CUDA Kernels

**Implemented Kernels**:

1. **`KernelComputeInternalForces`**
   - Computes element strains from displacements
   - Calculates stresses using material model
   - Assembles internal forces
   - Uses atomic operations for thread-safe assembly

2. **`KernelComputeMassMatrix`**
   - Lumped mass matrix assembly
   - Parallel computation over all elements

3. **`KernelComputeAccelerations`**
   - F = ma → a = F/m
   - Includes damping forces
   - Parallel over all nodes

4. **`KernelUpdateVelocities`**
   - v^(n+1) = v^n + a·dt
   - Central difference integration

5. **`KernelUpdateDisplacements`**
   - u^(n+1) = u^n + v·dt
   - Parallel update

6. **`KernelApplyBoundaryConditions`**
   - Enforces displacement/velocity constraints
   - Parallel over all BC nodes

**Kernel Launch Configuration**:
- Thread block size: 256 threads
- Dynamic grid sizing based on problem size
- Optimized for NVIDIA GPUs (compute capability 6.0+)

### 3. Build System Integration

**Updated Files**:
- `CMakeLists.txt` (root) - Added CUDA language support
- `src/core/CMakeLists.txt` - Conditional CUDA compilation

**New CMake Options**:
```cmake
option(FEMML_USE_GPU "Enable GPU acceleration" ON)
```

**CMake Detection**:
- Auto-detects CUDA Toolkit
- Sets CUDA architectures (60, 61, 70, 75, 80, 86)
- Links CUDA runtime library
- Defines `FEMML_USE_GPU` preprocessor macro

### 4. Python Bindings

**Added to** `src/python/api/femml_bindings.cpp`:

```python
import femml

# Create GPU solver
solver = femml.GPUExplicitSolver(mesh)

# GPU-specific methods
solver.set_device_id(0)              # Select GPU
gpu_mem = solver.get_gpu_memory_usage()  # Monitor memory
gpu_time = solver.get_gpu_compute_time() # Get compute time
transfer = solver.get_cpu_gpu_transfer_time()  # Transfer overhead
```

### 5. Example Script

**File**: `examples/job1_gpu/run_job1_gpu.py`

**Features**:
- Loads Job-1.inp mesh (29K nodes, 21K elements)
- Configures 1-second simulation
- Auto-calculates time step from CFL condition
- Real-time progress monitoring
- Performance metrics
- Result visualization

**Workflow**:
```python
1. Load mesh from Abaqus .inp file
2. Create aluminum material
3. Initialize GPU solver
4. Apply boundary conditions (fixed left end)
5. Apply tensile load (right end)
6. Configure solver parameters
7. Run GPU simulation
8. Export results (CSV + plots)
```

## GPU Memory Layout

### Device Memory Allocations

For a mesh with N nodes and E elements:

```
d_displacements_     : 3N × double  (u_x, u_y, u_z for each node)
d_velocities_        : 3N × double  (v_x, v_y, v_z for each node)
d_accelerations_     : 3N × double  (a_x, a_y, a_z for each node)
d_forces_internal_   : 3N × double  (f_int_x, f_int_y, f_int_z)
d_forces_external_   : 3N × double  (f_ext_x, f_ext_y, f_ext_z)
d_masses_            : N × double   (lumped mass per node)
d_element_nodes_     : 8E × int     (connectivity for C3D8)
d_node_coords_       : 3N × double  (x, y, z for each node)
d_material_props_    : 3 × double   (E, ν, ρ)
d_bc_nodes_          : NBC × int    (BC node IDs)
d_bc_components_     : NBC × int    (BC components)
```

**Total Memory for Job-1**:
- Nodes: 29,105
- Elements: 21,178
- Estimated GPU memory: ~480 MB

### Memory Transfer Pattern

```
Initialization:
  CPU → GPU: node coords, connectivity, material properties, BCs

During Solve:
  GPU only (no CPU-GPU transfer)

After Solve:
  GPU → CPU: displacements, velocities
```

## Performance Characteristics

### Computational Complexity

- **Element assembly**: O(E) - fully parallel
- **Time integration**: O(N) - fully parallel
- **Boundary conditions**: O(NBC) - fully parallel

### Expected Speedup

Compared to CPU (16-core):

| Mesh Size     | Elements | GPU Speedup |
|--------------|----------|-------------|
| Small        | 1K       | 5-10x       |
| Medium       | 10K      | 15-25x      |
| Large        | 100K     | 30-50x      |
| **Job-1**    | **21K**  | **20-40x**  |

### Bottlenecks

1. **Atomic operations** in force assembly
   - Necessary for thread safety
   - Can cause contention on shared nodes

2. **Data transfer**
   - Minimized by keeping data on GPU
   - Only transfer results at end

3. **Small kernels**
   - Update kernels are memory-bound
   - GPU may not be fully utilized

## Algorithm Implementation

### Explicit Time Integration

```
For each time step n:

  1. Compute element strains:
     ε = B·u^n
     [GPU Kernel: KernelComputeInternalForces]

  2. Compute element stresses:
     σ = C·ε  (linear elastic)
     [GPU Kernel: KernelComputeInternalForces]

  3. Assemble internal forces:
     f_int = Σ(B^T·σ·V)
     [GPU Kernel: KernelComputeInternalForces with atomicAdd]

  4. Compute accelerations:
     a^n = (f_ext - f_int - c·v^n) / m
     [GPU Kernel: KernelComputeAccelerations]

  5. Update velocities:
     v^(n+1) = v^n + a^n·dt
     [GPU Kernel: KernelUpdateVelocities]

  6. Update displacements:
     u^(n+1) = u^n + v^(n+1)·dt
     [GPU Kernel: KernelUpdateDisplacements]

  7. Enforce boundary conditions:
     u_bc = 0, v_bc = 0, a_bc = 0
     [GPU Kernel: KernelApplyBoundaryConditions]
```

### Hexahedral Element Formulation

**Shape Functions** (trilinear):
```
N_i(ξ,η,ζ) = 1/8·(1 + ξ_i·ξ)·(1 + η_i·η)·(1 + ζ_i·ζ)
```

**B-Matrix** (6×24):
```
B = [∂N_1/∂x    0         0      ...  ∂N_8/∂x    0         0     ]
    [0          ∂N_1/∂y   0      ...  0          ∂N_8/∂y   0     ]
    [0          0         ∂N_1/∂z ...  0          0         ∂N_8/∂z]
    [∂N_1/∂y    ∂N_1/∂x   0      ...  ∂N_8/∂y    ∂N_8/∂x   0     ]
    [∂N_1/∂z    0         ∂N_1/∂x ...  ∂N_8/∂z    0         ∂N_8/∂x]
    [0          ∂N_1/∂z   ∂N_1/∂y ...  0          ∂N_8/∂z   ∂N_8/∂y]
```

**Strain Calculation**:
```
ε = B·u_element
```

**Stress Calculation** (linear elastic):
```
σ_xx = λ·trace(ε) + 2μ·ε_xx
σ_yy = λ·trace(ε) + 2μ·ε_yy
σ_zz = λ·trace(ε) + 2μ·ε_zz
τ_xy = μ·γ_xy
τ_xz = μ·γ_xz
τ_yz = μ·γ_yz

where:
  λ = E·ν / ((1+ν)·(1-2ν))
  μ = E / (2·(1+ν))
```

**Internal Force**:
```
f_element = -B^T·σ·V
```

## Build Requirements

### Minimum Requirements

- **CUDA Toolkit**: 11.0 or later
- **GPU**: Compute capability 6.0+ (Pascal architecture)
  - Examples: GTX 1060, RTX 2060, RTX 3060, etc.
- **GPU Memory**: 512 MB minimum (4 GB recommended)
- **CMake**: 3.12+
- **Compiler**: MSVC 2019+ or GCC 7+ with CUDA support

### Recommended Configuration

- **GPU**: RTX 3080 or better (10+ GB VRAM)
- **CUDA**: 12.0+
- **RAM**: 16 GB system memory
- **OS**: Windows 10/11 or Linux

## Known Limitations

### Current Implementation

1. **Material Models**:
   - Only LinearElastic fully supported on GPU
   - NeuralMaterial would require callback execution on GPU (future work)

2. **Element Types**:
   - Currently only C3D8 (hexahedral) implemented
   - C3D4 (tetrahedral) planned

3. **Boundary Conditions**:
   - Fixed and prescribed displacement supported
   - Velocity/acceleration BCs need testing

4. **Load Types**:
   - Nodal forces supported
   - Pressure/body force loads not yet implemented

### Future Enhancements

1. **Multi-GPU Support**:
   - Domain decomposition
   - MPI + CUDA for distributed memory

2. **Adaptive Time Stepping**:
   - Dynamic dt based on CFL condition
   - Currently uses fixed dt

3. **Advanced Materials**:
   - Plasticity models on GPU
   - Neural network inference kernels

4. **Contact Detection**:
   - GPU-accelerated collision detection
   - Contact force computation

5. **Output Optimization**:
   - Direct GPU-to-VTK export
   - Avoid CPU transfer for visualization

## Debugging Tips

### CUDA Error Checking

All CUDA calls wrapped with `CUDA_CHECK` macro:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)
```

### Profiling

**NVIDIA Nsight**:
```powershell
nsys profile --trace=cuda python run_job1_gpu.py
```

**Memory Checking**:
```powershell
cuda-memcheck python run_job1_gpu.py
```

### Common Issues

**Issue**: `cudaMalloc failed: out of memory`
**Fix**: Reduce mesh size or use GPU with more memory

**Issue**: Kernel launch timeout
**Fix**: Increase Windows TDR timeout or use smaller time steps

**Issue**: Slow performance
**Fix**: Check GPU utilization with `nvidia-smi`, ensure proper CUDA architecture

## Testing Checklist

Before running on Job-1:

- [ ] CUDA Toolkit installed and in PATH
- [ ] GPU detected by system (`nvidia-smi`)
- [ ] FEM-ML built with `FEMML_USE_GPU=ON`
- [ ] Python femml module imports successfully
- [ ] `hasattr(femml, 'GPUExplicitSolver')` returns True
- [ ] GPU has sufficient memory (>500 MB free)
- [ ] Simple test case runs (single element)

## References

- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/
- Explicit FEM: Hughes, "The Finite Element Method"
- GPU acceleration: Nvidia GPU Gems series

---

**Created**: November 2024
**Version**: 1.0
**Status**: Ready for testing (requires CUDA hardware)
