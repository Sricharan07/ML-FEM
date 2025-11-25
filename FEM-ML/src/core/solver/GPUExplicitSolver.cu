#include "solver/GPUExplicitSolver.hpp"
#include "material/LinearElastic.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

namespace femml {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Compute B-matrix and internal forces for C3D8 hex element
 */
__device__ void ComputeHexElement(
    const double* node_coords,  // 8 nodes × 3 coords
    const double* displacement,  // 8 nodes × 3 dofs
    double E, double nu, double rho,
    double* force_out           // 8 nodes × 3 forces
) {
    // Material properties (Lamé parameters)
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));

    // Shape function derivatives at center (ξ=η=ζ=0)
    double dN_dxi[8] = {-0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.125};
    double dN_deta[8] = {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125, 0.125};
    double dN_dzeta[8] = {-0.125, -0.125, -0.125, -0.125, 0.125, 0.125, 0.125, 0.125};

    // Compute Jacobian
    double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < 8; i++) {
        J[0][0] += dN_dxi[i] * node_coords[i * 3 + 0];
        J[0][1] += dN_dxi[i] * node_coords[i * 3 + 1];
        J[0][2] += dN_dxi[i] * node_coords[i * 3 + 2];

        J[1][0] += dN_deta[i] * node_coords[i * 3 + 0];
        J[1][1] += dN_deta[i] * node_coords[i * 3 + 1];
        J[1][2] += dN_deta[i] * node_coords[i * 3 + 2];

        J[2][0] += dN_dzeta[i] * node_coords[i * 3 + 0];
        J[2][1] += dN_dzeta[i] * node_coords[i * 3 + 1];
        J[2][2] += dN_dzeta[i] * node_coords[i * 3 + 2];
    }

    // Compute determinant
    double detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
                - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
                + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    double volume = detJ * 8.0; // For single-point integration

    // Inverse Jacobian
    double invJ[3][3];
    double inv_detJ = 1.0 / detJ;
    invJ[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv_detJ;
    invJ[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * inv_detJ;
    invJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv_detJ;
    invJ[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * inv_detJ;
    invJ[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv_detJ;
    invJ[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * inv_detJ;
    invJ[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv_detJ;
    invJ[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * inv_detJ;
    invJ[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv_detJ;

    // Compute shape function derivatives in global coordinates
    double dN_dx[8], dN_dy[8], dN_dz[8];
    for (int i = 0; i < 8; i++) {
        dN_dx[i] = invJ[0][0] * dN_dxi[i] + invJ[0][1] * dN_deta[i] + invJ[0][2] * dN_dzeta[i];
        dN_dy[i] = invJ[1][0] * dN_dxi[i] + invJ[1][1] * dN_deta[i] + invJ[1][2] * dN_dzeta[i];
        dN_dz[i] = invJ[2][0] * dN_dxi[i] + invJ[2][1] * dN_deta[i] + invJ[2][2] * dN_dzeta[i];
    }

    // Compute strain
    double strain[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        strain[0] += dN_dx[i] * displacement[i * 3 + 0]; // εxx
        strain[1] += dN_dy[i] * displacement[i * 3 + 1]; // εyy
        strain[2] += dN_dz[i] * displacement[i * 3 + 2]; // εzz
        strain[3] += dN_dx[i] * displacement[i * 3 + 1] + dN_dy[i] * displacement[i * 3 + 0]; // γxy
        strain[4] += dN_dx[i] * displacement[i * 3 + 2] + dN_dz[i] * displacement[i * 3 + 0]; // γxz
        strain[5] += dN_dy[i] * displacement[i * 3 + 2] + dN_dz[i] * displacement[i * 3 + 1]; // γyz
    }

    // Compute stress (linear elastic)
    double trace = strain[0] + strain[1] + strain[2];
    double stress[6];
    stress[0] = lambda * trace + 2.0 * mu * strain[0];
    stress[1] = lambda * trace + 2.0 * mu * strain[1];
    stress[2] = lambda * trace + 2.0 * mu * strain[2];
    stress[3] = mu * strain[3];
    stress[4] = mu * strain[4];
    stress[5] = mu * strain[5];

    // Compute internal forces: f = B^T * σ * V
    for (int i = 0; i < 8; i++) {
        force_out[i * 3 + 0] = -(dN_dx[i] * stress[0] + dN_dy[i] * stress[3] + dN_dz[i] * stress[4]) * volume;
        force_out[i * 3 + 1] = -(dN_dx[i] * stress[3] + dN_dy[i] * stress[1] + dN_dz[i] * stress[5]) * volume;
        force_out[i * 3 + 2] = -(dN_dx[i] * stress[4] + dN_dy[i] * stress[5] + dN_dz[i] * stress[2]) * volume;
    }
}

/**
 * @brief Kernel: Compute internal forces for all elements
 */
__global__ void KernelComputeInternalForces(
    const double* d_displacements,
    const double* d_node_coords,
    const int* d_element_nodes,
    const double* d_material_props,  // [E, nu, rho]
    double* d_forces_internal,
    int num_elements,
    int num_nodes
) {
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements) return;

    // Get material properties
    double E = d_material_props[0];
    double nu = d_material_props[1];
    double rho = d_material_props[2];

    // Get element nodes
    int nodes[8];
    double elem_coords[24];  // 8 nodes × 3
    double elem_disp[24];    // 8 nodes × 3

    for (int i = 0; i < 8; i++) {
        nodes[i] = d_element_nodes[elem_idx * 8 + i];

        elem_coords[i * 3 + 0] = d_node_coords[nodes[i] * 3 + 0];
        elem_coords[i * 3 + 1] = d_node_coords[nodes[i] * 3 + 1];
        elem_coords[i * 3 + 2] = d_node_coords[nodes[i] * 3 + 2];

        elem_disp[i * 3 + 0] = d_displacements[nodes[i] * 3 + 0];
        elem_disp[i * 3 + 1] = d_displacements[nodes[i] * 3 + 1];
        elem_disp[i * 3 + 2] = d_displacements[nodes[i] * 3 + 2];
    }

    // Compute element forces
    double elem_forces[24];
    ComputeHexElement(elem_coords, elem_disp, E, nu, rho, elem_forces);

    // Assemble to global force vector (atomic add for thread safety)
    for (int i = 0; i < 8; i++) {
        atomicAdd(&d_forces_internal[nodes[i] * 3 + 0], elem_forces[i * 3 + 0]);
        atomicAdd(&d_forces_internal[nodes[i] * 3 + 1], elem_forces[i * 3 + 1]);
        atomicAdd(&d_forces_internal[nodes[i] * 3 + 2], elem_forces[i * 3 + 2]);
    }
}

/**
 * @brief Kernel: Compute lumped mass matrix
 */
__global__ void KernelComputeMassMatrix(
    const double* d_node_coords,
    const int* d_element_nodes,
    const double* d_material_props,
    double* d_masses,
    int num_elements,
    int num_nodes
) {
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_idx >= num_elements) return;

    double rho = d_material_props[2];

    // Get element coordinates
    double elem_coords[24];
    int nodes[8];
    for (int i = 0; i < 8; i++) {
        nodes[i] = d_element_nodes[elem_idx * 8 + i];
        elem_coords[i * 3 + 0] = d_node_coords[nodes[i] * 3 + 0];
        elem_coords[i * 3 + 1] = d_node_coords[nodes[i] * 3 + 1];
        elem_coords[i * 3 + 2] = d_node_coords[nodes[i] * 3 + 2];
    }

    // Compute element volume (approximate)
    double dx = elem_coords[3] - elem_coords[0];
    double dy = elem_coords[9] - elem_coords[0];
    double dz = elem_coords[12] - elem_coords[0];
    double volume = fabs(dx * dy * dz);

    // Lumped mass per node
    double node_mass = rho * volume / 8.0;

    // Add to global mass vector
    for (int i = 0; i < 8; i++) {
        atomicAdd(&d_masses[nodes[i]], node_mass);
    }
}

/**
 * @brief Kernel: Compute accelerations
 */
__global__ void KernelComputeAccelerations(
    double* d_accelerations,
    const double* d_forces_internal,
    const double* d_forces_external,
    const double* d_masses,
    const double* d_velocities,
    double damping,
    int num_nodes
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;

    double mass = d_masses[node_idx];
    if (mass < 1e-12) return; // Avoid division by zero

    for (int i = 0; i < 3; i++) {
        int dof = node_idx * 3 + i;
        double f_net = d_forces_external[dof] - d_forces_internal[dof] - damping * mass * d_velocities[dof];
        d_accelerations[dof] = f_net / mass;
    }
}

/**
 * @brief Kernel: Update velocities
 */
__global__ void KernelUpdateVelocities(
    double* d_velocities,
    const double* d_accelerations,
    double dt,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dofs = num_nodes * 3;
    if (idx >= total_dofs) return;

    d_velocities[idx] += d_accelerations[idx] * dt;
}

/**
 * @brief Kernel: Update displacements
 */
__global__ void KernelUpdateDisplacements(
    double* d_displacements,
    const double* d_velocities,
    double dt,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dofs = num_nodes * 3;
    if (idx >= total_dofs) return;

    d_displacements[idx] += d_velocities[idx] * dt;
}

/**
 * @brief Kernel: Apply boundary conditions
 */
__global__ void KernelApplyBoundaryConditions(
    double* d_displacements,
    double* d_velocities,
    double* d_accelerations,
    const int* d_bc_nodes,
    const int* d_bc_components,
    int num_constraints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_constraints) return;

    int node_id = d_bc_nodes[idx];
    int component = d_bc_components[idx];

    if (component == -1) {
        // Fix all components
        for (int i = 0; i < 3; i++) {
            d_displacements[node_id * 3 + i] = 0.0;
            d_velocities[node_id * 3 + i] = 0.0;
            d_accelerations[node_id * 3 + i] = 0.0;
        }
    } else {
        // Fix specific component
        d_displacements[node_id * 3 + component] = 0.0;
        d_velocities[node_id * 3 + component] = 0.0;
        d_accelerations[node_id * 3 + component] = 0.0;
    }
}

// ============================================================================
// Kernel Launch Functions (C interface)
// ============================================================================

extern "C" {

void LaunchComputeInternalForces(
    const double* d_displacements,
    const double* d_node_coords,
    const int* d_element_nodes,
    const double* d_material_props,
    double* d_forces_internal,
    int num_elements,
    int num_nodes
) {
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Zero out forces first
    CUDA_CHECK(cudaMemset(d_forces_internal, 0, num_nodes * 3 * sizeof(double)));

    KernelComputeInternalForces<<<num_blocks, threads_per_block>>>(
        d_displacements, d_node_coords, d_element_nodes, d_material_props,
        d_forces_internal, num_elements, num_nodes
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LaunchComputeMassMatrix(
    const double* d_node_coords,
    const int* d_element_nodes,
    const double* d_material_props,
    double* d_masses,
    int num_elements,
    int num_nodes
) {
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemset(d_masses, 0, num_nodes * sizeof(double)));

    KernelComputeMassMatrix<<<num_blocks, threads_per_block>>>(
        d_node_coords, d_element_nodes, d_material_props,
        d_masses, num_elements, num_nodes
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LaunchUpdateVelocities(
    double* d_velocities,
    const double* d_accelerations,
    double dt,
    int num_nodes
) {
    int total_dofs = num_nodes * 3;
    int threads_per_block = 256;
    int num_blocks = (total_dofs + threads_per_block - 1) / threads_per_block;

    KernelUpdateVelocities<<<num_blocks, threads_per_block>>>(
        d_velocities, d_accelerations, dt, num_nodes
    );

    CUDA_CHECK(cudaGetLastError());
}

void LaunchUpdateDisplacements(
    double* d_displacements,
    const double* d_velocities,
    double dt,
    int num_nodes
) {
    int total_dofs = num_nodes * 3;
    int threads_per_block = 256;
    int num_blocks = (total_dofs + threads_per_block - 1) / threads_per_block;

    KernelUpdateDisplacements<<<num_blocks, threads_per_block>>>(
        d_displacements, d_velocities, dt, num_nodes
    );

    CUDA_CHECK(cudaGetLastError());
}

void LaunchComputeAccelerations(
    double* d_accelerations,
    const double* d_forces_internal,
    const double* d_forces_external,
    const double* d_masses,
    const double* d_velocities,
    double damping,
    int num_nodes
) {
    int threads_per_block = 256;
    int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    KernelComputeAccelerations<<<num_blocks, threads_per_block>>>(
        d_accelerations, d_forces_internal, d_forces_external,
        d_masses, d_velocities, damping, num_nodes
    );

    CUDA_CHECK(cudaGetLastError());
}

void LaunchApplyBoundaryConditions(
    double* d_displacements,
    double* d_velocities,
    double* d_accelerations,
    const int* d_bc_nodes,
    const int* d_bc_components,
    int num_constraints
) {
    int threads_per_block = 256;
    int num_blocks = (num_constraints + threads_per_block - 1) / threads_per_block;

    KernelApplyBoundaryConditions<<<num_blocks, threads_per_block>>>(
        d_displacements, d_velocities, d_accelerations,
        d_bc_nodes, d_bc_components, num_constraints
    );

    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"

// ============================================================================
// GPUExplicitSolver Implementation
// ============================================================================

GPUExplicitSolver::GPUExplicitSolver(const std::shared_ptr<Mesh>& mesh)
    : ExplicitSolver(mesh) {
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable GPU found!");
    }

    std::cout << "GPU Solver initialized with " << device_count << " CUDA device(s)" << std::endl;
}

GPUExplicitSolver::~GPUExplicitSolver() {
    FreeGPUMemory();
}

void GPUExplicitSolver::SetDeviceID(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    device_id_ = device_id;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
}

void GPUExplicitSolver::AllocateGPUMemory() {
    int num_nodes = mesh_->GetNumNodes();
    int num_elements = mesh_->GetNumElements();

    std::cout << "Allocating GPU memory for " << num_nodes << " nodes, "
              << num_elements << " elements..." << std::endl;

    // Allocate arrays
    CUDA_CHECK(cudaMalloc(&d_displacements_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velocities_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_accelerations_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces_internal_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_forces_external_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_masses_, num_nodes * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&d_element_nodes_, num_elements * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_coords_, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_material_props_, 3 * sizeof(double)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_displacements_, 0, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_velocities_, 0, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_accelerations_, 0, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forces_internal_, 0, num_nodes * 3 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_forces_external_, 0, num_nodes * 3 * sizeof(double)));

    std::cout << "GPU memory allocated successfully" << std::endl;
}

void GPUExplicitSolver::FreeGPUMemory() {
    if (d_displacements_) cudaFree(d_displacements_);
    if (d_velocities_) cudaFree(d_velocities_);
    if (d_accelerations_) cudaFree(d_accelerations_);
    if (d_forces_internal_) cudaFree(d_forces_internal_);
    if (d_forces_external_) cudaFree(d_forces_external_);
    if (d_masses_) cudaFree(d_masses_);
    if (d_element_nodes_) cudaFree(d_element_nodes_);
    if (d_node_coords_) cudaFree(d_node_coords_);
    if (d_material_props_) cudaFree(d_material_props_);
    if (d_bc_nodes_) cudaFree(d_bc_nodes_);
    if (d_bc_components_) cudaFree(d_bc_components_);
}

void GPUExplicitSolver::CopyDataToGPU() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_nodes = mesh_->GetNumNodes();
    int num_elements = mesh_->GetNumElements();

    // Copy node coordinates
    std::vector<double> coords(num_nodes * 3);
    for (int i = 0; i < num_nodes; i++) {
        auto node = mesh_->GetNode(i + 1);
        coords[i * 3 + 0] = node.coords[0];
        coords[i * 3 + 1] = node.coords[1];
        coords[i * 3 + 2] = node.coords[2];
    }
    CUDA_CHECK(cudaMemcpy(d_node_coords_, coords.data(), num_nodes * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // Copy element connectivity
    std::vector<int> connectivity(num_elements * 8);
    for (int i = 0; i < num_elements; i++) {
        auto elem = mesh_->GetElement(i + 1);
        for (int j = 0; j < 8; j++) {
            connectivity[i * 8 + j] = elem.nodes[j] - 1; // Convert to 0-based
        }
    }
    CUDA_CHECK(cudaMemcpy(d_element_nodes_, connectivity.data(), num_elements * 8 * sizeof(int), cudaMemcpyHostToDevice));

    // Copy material properties (use "ALL" material if available)
    if (materials_.empty()) {
        throw std::runtime_error("GPUExplicitSolver requires at least one material before initialization");
    }

    std::shared_ptr<Material> material = nullptr;
    auto mat_it = materials_.find("ALL");
    if (mat_it != materials_.end()) {
        material = mat_it->second;
    } else {
        material = materials_.begin()->second;
    }

    if (!material) {
        throw std::runtime_error("GPUExplicitSolver received a null material reference");
    }

    auto linear_material = std::dynamic_pointer_cast<LinearElastic>(material);
    if (!linear_material) {
        throw std::runtime_error("GPUExplicitSolver currently supports only LinearElastic materials");
    }

    double mat_props[3] = {
        linear_material->GetYoungsModulus(),
        linear_material->GetPoissonsRatio(),
        material->GetDensity()
    };
    CUDA_CHECK(cudaMemcpy(d_material_props_, mat_props, 3 * sizeof(double), cudaMemcpyHostToDevice));

    auto end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double>(end - start).count();
}

void GPUExplicitSolver::CopyDataFromGPU() {
    auto start = std::chrono::high_resolution_clock::now();

    int num_nodes = mesh_->GetNumNodes();

    // Copy displacements back
    std::vector<double> disp(num_nodes * 3);
    CUDA_CHECK(cudaMemcpy(disp.data(), d_displacements_, num_nodes * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_nodes; i++) {
        displacements_[i][0] = disp[i * 3 + 0];
        displacements_[i][1] = disp[i * 3 + 1];
        displacements_[i][2] = disp[i * 3 + 2];
    }

    // Copy velocities
    std::vector<double> vel(num_nodes * 3);
    CUDA_CHECK(cudaMemcpy(vel.data(), d_velocities_, num_nodes * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_nodes; i++) {
        velocities_[i][0] = vel[i * 3 + 0];
        velocities_[i][1] = vel[i * 3 + 1];
        velocities_[i][2] = vel[i * 3 + 2];
    }

    auto end = std::chrono::high_resolution_clock::now();
    transfer_time_ += std::chrono::duration<double>(end - start).count();
}

void GPUExplicitSolver::Initialize() {
    std::cout << "\n=== GPU Explicit Solver Initialization ===" << std::endl;

    // Call base class initialization
    ExplicitSolver::Initialize();

    // Set default GPU
    SetDeviceID(0);

    // Allocate GPU memory
    AllocateGPUMemory();

    // Copy data to GPU
    CopyDataToGPU();

    // Compute mass matrix on GPU
    ComputeMassMatrixGPU();

    // Prepare boundary conditions
    std::vector<int> bc_nodes_host, bc_components_host;
    for (const auto& bc : boundary_conditions_) {
        for (int node_id : bc.nodes) {
            bc_nodes_host.push_back(node_id - 1); // 0-based
            bc_components_host.push_back(bc.component);
        }
    }

    num_bc_constraints_ = bc_nodes_host.size();
    if (num_bc_constraints_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_bc_nodes_, num_bc_constraints_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bc_components_, num_bc_constraints_ * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bc_nodes_, bc_nodes_host.data(), num_bc_constraints_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bc_components_, bc_components_host.data(), num_bc_constraints_ * sizeof(int), cudaMemcpyHostToDevice));
    }

    std::cout << "GPU initialization complete!" << std::endl;
    std::cout << "  GPU Memory Usage: " << GetGPUMemoryUsage() / (1024 * 1024) << " MB" << std::endl;
}

void GPUExplicitSolver::ComputeMassMatrixGPU() {
    LaunchComputeMassMatrix(
        d_node_coords_,
        d_element_nodes_,
        d_material_props_,
        d_masses_,
        mesh_->GetNumElements(),
        mesh_->GetNumNodes()
    );
}

void GPUExplicitSolver::Step() {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Compute internal forces
    LaunchComputeInternalForces(
        d_displacements_,
        d_node_coords_,
        d_element_nodes_,
        d_material_props_,
        d_forces_internal_,
        mesh_->GetNumElements(),
        mesh_->GetNumNodes()
    );

    // 2. Compute accelerations
    LaunchComputeAccelerations(
        d_accelerations_,
        d_forces_internal_,
        d_forces_external_,
        d_masses_,
        d_velocities_,
        params_.damping,
        mesh_->GetNumNodes()
    );

    // 3. Update velocities
    LaunchUpdateVelocities(d_velocities_, d_accelerations_, params_.time_step, mesh_->GetNumNodes());

    // 4. Update displacements
    LaunchUpdateDisplacements(d_displacements_, d_velocities_, params_.time_step, mesh_->GetNumNodes());

    // 5. Apply boundary conditions
    if (num_bc_constraints_ > 0) {
        LaunchApplyBoundaryConditions(
            d_displacements_,
            d_velocities_,
            d_accelerations_,
            d_bc_nodes_,
            d_bc_components_,
            num_bc_constraints_
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    current_time_ += params_.time_step;
    current_step_++;

    auto end = std::chrono::high_resolution_clock::now();
    gpu_compute_time_ += std::chrono::duration<double>(end - start).count();
}

void GPUExplicitSolver::Solve() {
    std::cout << "\n=== Starting GPU Simulation ===" << std::endl;
    std::cout << "Time steps: " << params_.num_steps << std::endl;
    std::cout << "Time step size: " << params_.time_step << " s" << std::endl;

    auto solve_start = std::chrono::high_resolution_clock::now();

    while (current_step_ < params_.num_steps) {
        Step();

        if (current_step_ % params_.output_interval == 0) {
            if (output_callback_) {
                output_callback_(current_step_, current_time_);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to CPU
    CopyDataFromGPU();
    // Refresh CPU-side element stresses/strains for output
    ComputeInternalForces();

    auto solve_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(solve_end - solve_start).count();

    std::cout << "\n=== GPU Simulation Complete ===" << std::endl;
    std::cout << "Total time: " << total_time << " s" << std::endl;
    std::cout << "GPU compute time: " << gpu_compute_time_ << " s" << std::endl;
    std::cout << "Data transfer time: " << transfer_time_ << " s" << std::endl;
    std::cout << "Performance: " << params_.num_steps / total_time << " steps/s" << std::endl;
}

size_t GPUExplicitSolver::GetGPUMemoryUsage() const {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem - free_mem;
}

} // namespace femml
