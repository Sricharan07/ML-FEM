# Job-1 GPU Quick Start

## Prerequisites

1. **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
2. **CUDA Toolkit** 11.0 or later
3. **Python** 3.7+ with numpy, matplotlib

## Installation

### Step 1: Install CUDA Toolkit

Download and install from: https://developer.nvidia.com/cuda-downloads

Verify installation:
```powershell
nvcc --version
nvidia-smi
```

### Step 2: Build FEM-ML with GPU Support

```powershell
cd D:/Research-work/ML-FEM/FEM-ML

# Clean previous build (if exists)
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# Build with GPU enabled
.\build.ps1
```

The build script will auto-detect CUDA and enable GPU support.

### Step 3: Verify GPU Build

```powershell
# Check if GPU solver is available
python -c "import sys; sys.path.insert(0, 'build/src/python/Release'); import femml; print('GPU Available:', hasattr(femml, 'GPUExplicitSolver'))"
```

Expected output: `GPU Available: True`

## Running Job-1

### Quick Run (1 second simulation)

```powershell
cd examples/job1_gpu
python run_job1_gpu.py
```

### Expected Runtime

| GPU Model    | Approx. Time |
|-------------|--------------|
| RTX 4090    | ~65 seconds  |
| RTX 3080    | ~115 seconds |
| RTX 2060    | ~210 seconds |
| GTX 1660    | ~290 seconds |

### Output Files

After completion, you'll find:
- `job1_results.csv` - Nodal displacements
- `job1_displacement_magnitude.txt` - Displacement magnitudes
- `job1_results.png` - Visualization plots

## Troubleshooting

### Build Fails: "CUDA not found"

**Solution**:
1. Install CUDA Toolkit
2. Add to PATH:
   ```powershell
   $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
   ```
3. Rebuild

### Import Error: "femml module not found"

**Solution**:
```powershell
# Add build directory to Python path
$env:PYTHONPATH = "D:/Research-work/ML-FEM/FEM-ML/build/src/python\Release"
python run_job1_gpu.py
```

### Runtime Error: "Out of GPU memory"

**Solution**:
1. Close other GPU applications
2. Reduce simulation steps:
   ```python
   # In run_job1_gpu.py, change:
   total_time = 0.5  # Instead of 1.0
   ```

### Slow Performance

**Check GPU utilization**:
```powershell
nvidia-smi -l 1
```

Should show GPU at ~90-100% utilization during simulation.

## Quick Test (30 seconds)

Before running full Job-1, test with reduced steps:

```python
# Edit run_job1_gpu.py line ~180:
total_time = 0.1  # 0.1 second instead of 1.0
```

This will complete in ~20-30 seconds and verify everything works.

## Next Steps

- Check [README.md](README.md) for detailed usage
- See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details
- Read [FEM-ML Documentation](../../docs/USER_GUIDE.md) for more examples

## Support

If you encounter issues:
1. Check CUDA installation: `nvidia-smi`
2. Verify GPU compute capability: Should be 6.0 or higher
3. Check build logs for CUDA-related errors
4. Ensure sufficient GPU memory (>512 MB free)

---

**Estimated Total Time: 15-30 minutes** (install CUDA + build + first run)
