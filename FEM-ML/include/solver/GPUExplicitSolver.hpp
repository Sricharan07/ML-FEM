#pragma once

#include "solver/ExplicitSolver.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace femml {

/**
 * @brief GPU-accelerated Explicit Dynamics Solver using CUDA
 *
 * Accelerates computation of:
 * - Element internal forces
 * - Mass matrix assembly
 * - Time integration
 * - Boundary condition enforcement
 */
class GPUExplicitSolver : public ExplicitSolver {
public:
    GPUExplicitSolver(const std::shared_ptr<Mesh>& mesh);
    ~GPUExplicitSolver() override;

    // Override solver methods to use GPU
    void Initialize() override;
    void Step() override;
    void Solve() override;

    // GPU-specific methods
    void SetDeviceID(int device_id);
    int GetDeviceID() const { return device_id_; }

    // Performance metrics
    double GetGPUComputeTime() const { return gpu_compute_time_; }
    double GetCPUGPUTransferTime() const { return transfer_time_; }
    size_t GetGPUMemoryUsage() const;

private:
    // GPU device management
    int device_id_ = 0;

    // Device memory pointers
    double* d_displacements_ = nullptr;     // 3 * num_nodes
    double* d_velocities_ = nullptr;        // 3 * num_nodes
    double* d_accelerations_ = nullptr;     // 3 * num_nodes
    double* d_forces_internal_ = nullptr;   // 3 * num_nodes
    double* d_forces_external_ = nullptr;   // 3 * num_nodes
    double* d_masses_ = nullptr;            // num_nodes

    // Element data on GPU
    int* d_element_nodes_ = nullptr;        // 8 * num_elements (for C3D8)
    double* d_node_coords_ = nullptr;       // 3 * num_nodes
    double* d_material_props_ = nullptr;    // Material properties (E, nu, rho)

    // Boundary condition data
    int* d_bc_nodes_ = nullptr;             // BC node IDs
    int* d_bc_components_ = nullptr;        // BC components
    int num_bc_constraints_ = 0;

    // Performance tracking
    double gpu_compute_time_ = 0.0;
    double transfer_time_ = 0.0;

    // GPU initialization
    void AllocateGPUMemory();
    void FreeGPUMemory();
    void CopyDataToGPU();
    void CopyDataFromGPU();

    // GPU compute kernels (implemented in .cu file)
    void ComputeInternalForcesGPU();
    void ComputeMassMatrixGPU();
    void UpdateVelocitiesGPU(double dt);
    void UpdateDisplacementsGPU(double dt);
    void ApplyBoundaryConditionsGPU();
    void ApplyExternalLoadsGPU();
};

// CUDA kernel declarations (implemented in GPUExplicitSolver.cu)
extern "C" {
    void LaunchComputeInternalForces(
        const double* d_displacements,
        const double* d_node_coords,
        const int* d_element_nodes,
        const double* d_material_props,
        double* d_forces_internal,
        int num_elements,
        int num_nodes
    );

    void LaunchComputeMassMatrix(
        const double* d_node_coords,
        const int* d_element_nodes,
        const double* d_material_props,
        double* d_masses,
        int num_elements,
        int num_nodes
    );

    void LaunchUpdateVelocities(
        double* d_velocities,
        const double* d_accelerations,
        double dt,
        int num_nodes
    );

    void LaunchUpdateDisplacements(
        double* d_displacements,
        const double* d_velocities,
        double dt,
        int num_nodes
    );

    void LaunchComputeAccelerations(
        double* d_accelerations,
        const double* d_forces_internal,
        const double* d_forces_external,
        const double* d_masses,
        const double* d_velocities,
        double damping,
        int num_nodes
    );

    void LaunchApplyBoundaryConditions(
        double* d_displacements,
        double* d_velocities,
        double* d_accelerations,
        const int* d_bc_nodes,
        const int* d_bc_components,
        int num_constraints
    );
}

} // namespace femml
