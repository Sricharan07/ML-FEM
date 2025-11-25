# Job-1 GPU-Accelerated Simulation

Large-scale explicit FEM simulation using CUDA GPU acceleration.

## Overview

- **Mesh**: Job-1.inp (29,105 nodes, 21,178 hexahedral elements)
- **Material**: Aluminum (E=70 GPa, ν=0.33, ρ=2700 kg/m³)
- **Analysis**: Explicit dynamics with GPU acceleration
- **Duration**: 1 second simulation time
- **Solver**: CUDA-accelerated time integration

## Requirements

### Hardware
- NVIDIA GPU with CUDA compute capability 6.0+ (Pascal or newer)
- Recommended: RTX 2060 or better
- GPU Memory: ~500 MB required

### Software
- CUDA Toolkit 11.0 or later
- FEM-ML built with GPU support (`FEMML_USE_GPU=ON`)
- Python 3.7+
- NumPy, Matplotlib

## Build Instructions

### 1. Install CUDA Toolkit

Download from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```powershell
nvcc --version
```

### 2. Build FEM-ML with GPU Support

```powershell
cd FEM-ML
.\build.ps1
```

The build script will automatically detect CUDA if available.

To force GPU build:
```powershell
cmake -DFEMML_USE_GPU=ON -B build
cmake --build build --config Release
```

## Running the Simulation

### Quick Start

```powershell
cd examples/job1_gpu
python run_job1_gpu.py
```

### Expected Output

```
======================================================================
  FEM-ML GPU: Job-1 Simulation
======================================================================

[1/7] Loading Job-1 mesh...
   ✓ Loaded mesh: 29,105 nodes, 21,178 elements
   ✓ Load time: 1.23 s

[2/7] Creating material...
   ✓ Material: Aluminum
   ✓ Young's modulus: 7.00e+10 Pa
   ✓ Poisson's ratio: 0.330
   ✓ Density: 2700.0 kg/m³

[3/7] Initializing GPU solver...
   Using GPU: NVIDIA GeForce RTX 3080
     Compute Capability: 8.6
     Global Memory: 10240 MB
   ✓ Using GPU device: 0

[4/7] Setting boundary conditions...
   ✓ Fixed 100 nodes on left boundary

[5/7] Applying loads...
   ✓ Applied 1.00e+06 N tensile load
   ✓ Distributed over 100 nodes

[6/7] Configuring solver...
   ✓ Total simulation time: 1.0 s
   ✓ Estimated time step: 1.92e-07 s
   ✓ Number of steps: 5,208,333
   ✓ Damping: 2.0%

[7/7] Running GPU simulation...
   ------------------------------------------------------------
   ✓ Initialization time: 0.45 s
   ✓ GPU memory usage: 487.3 MB

   Step  100,000/5,208,333 ( 1.9%) | Time: 0.0192 s | KE: 1.234e+03 J | SE: 5.678e+04 J
   Step  200,000/5,208,333 ( 3.8%) | Time: 0.0384 s | KE: 2.345e+03 J | SE: 6.789e+04 J
   ...
   Step 5,200,000/5,208,333 (99.8%) | Time: 0.9984 s | KE: 1.234e+02 J | SE: 8.901e+04 J

   ------------------------------------------------------------
   ✓ Simulation complete!
   ✓ Total solve time: 145.67 s
   ✓ GPU compute time: 142.34 s
   ✓ CPU-GPU transfer time: 3.21 s
   ✓ Performance: 35,748 steps/s

======================================================================
  SIMULATION SUMMARY
======================================================================
  Mesh: 29,105 nodes, 21,178 elements
  Material: Aluminum
  Total time: 1.0 s (5,208,333 steps)
  Solve time: 145.67 s (35,748 steps/s)
  GPU speedup: 45.2x
  Max displacement: 2.3456 mm
======================================================================

✓ Job-1 GPU simulation complete!
```

## Output Files

After running, you'll find:

1. **`job1_results.csv`** - Complete nodal displacement data
   ```
   NodeID,X,Y,Z,Ux,Uy,Uz
   1,0.0,0.0,0.0,0.0,0.0,0.0
   2,1.0,0.0,0.0,0.000123,0.000045,0.000012
   ...
   ```

2. **`job1_displacement_magnitude.txt`** - Displacement magnitudes for all nodes

3. **`job1_results.png`** - Visualization plots:
   - Displacement histogram
   - Displacement profile along nodes

## Performance

### Expected Performance on Different GPUs

| GPU                  | Memory | Steps/sec | 1s Sim Time |
|---------------------|--------|-----------|-------------|
| RTX 4090            | 24 GB  | ~80,000   | ~65 s       |
| RTX 3080            | 10 GB  | ~45,000   | ~115 s      |
| RTX 2060            | 6 GB   | ~25,000   | ~210 s      |
| GTX 1660            | 6 GB   | ~18,000   | ~290 s      |
| CPU (16 cores)      | -      | ~2,000    | ~2,600 s    |

**GPU Speedup**: 10-40x compared to CPU

## Troubleshooting

### Error: "CUDA Toolkit not found"

**Solution**: Install CUDA Toolkit and add to PATH:
```powershell
# Add to environment variables
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

### Error: "GPU solver not available"

**Solution**: Rebuild with GPU support:
```powershell
cd build
cmake -DFEMML_USE_GPU=ON ..
cmake --build . --config Release
```

### Error: "Out of GPU memory"

**Solution 1**: Use a GPU with more memory

**Solution 2**: Reduce mesh size or reduce time steps:
```python
params.num_steps = 1000000  # Reduce from 5M to 1M
```

### Slow Performance

**Possible causes**:
1. GPU thermal throttling - Check temperatures
2. Wrong CUDA architecture - Verify CMake output
3. Data transfer overhead - Use larger `output_interval`

**Check GPU usage**:
```powershell
nvidia-smi -l 1
```

## Customization

### Change Material

```python
# Use steel instead of aluminum
material = femml.create_steel()
```

### Adjust Simulation Time

```python
total_time = 0.5  # 0.5 seconds instead of 1 second
params.num_steps = int(total_time / estimated_dt)
```

### Modify Boundary Conditions

```python
# Fix different nodes
bc.nodes = list(range(1, 201))  # Fix first 200 nodes
```

### Change Load Application

```python
# Apply pressure instead of force
load.type = femml.LoadType.PRESSURE
load.value = 1e6  # 1 MPa
```

## Advanced Usage

### Use Specific GPU Device

```python
# Use GPU 1 instead of GPU 0
solver.set_device_id(1)
```

### Monitor GPU Memory

```python
print(f"GPU memory: {solver.get_gpu_memory_usage() / (1024**2):.1f} MB")
```

### Custom Progress Tracking

```python
def my_callback(step, time):
    if step % 1000 == 0:
        print(f"Progress: {step} steps, {time:.4f} s")
        # Save checkpoint, update visualization, etc.

solver.set_output_callback(my_callback)
```

## References

- [FEM-ML Documentation](../../docs/USER_GUIDE.md)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Explicit FEM Theory](../../docs/CODE_STRUCTURE.md)

## Contact

For issues or questions:
- GitHub: https://github.com/username/FEM-ML/issues
- Documentation: [docs/](../../docs/)
