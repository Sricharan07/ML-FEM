#include "solver/ExplicitSolver.hpp"
#include "element/HexElement.hpp"
#include "element/TetElement.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iomanip>

namespace femml {

ExplicitSolver::ExplicitSolver(const std::shared_ptr<Mesh>& mesh)
    : mesh_(mesh), current_step_(0), current_time_(0.0), strain_energy_(0.0) {
}

void ExplicitSolver::SetMaterial(const std::string& element_set,
                                 const std::shared_ptr<Material>& material) {
    materials_[element_set] = material;
}

void ExplicitSolver::SetMaterial(const std::shared_ptr<Material>& material) {
    materials_["ALL"] = material;
}

void ExplicitSolver::AddBoundaryCondition(const BoundaryCondition& bc) {
    boundary_conditions_.push_back(bc);
}

void ExplicitSolver::AddLoad(const Load& load) {
    loads_.push_back(load);
}

void ExplicitSolver::SetParameters(const SolverParams& params) {
    params_ = params;
}

void ExplicitSolver::Initialize() {
    std::cout << "=== Initializing Explicit Solver ===" << std::endl;

    // Create mapping from node ID to array index
    int index = 0;
    for (const auto& [node_id, node] : mesh_->GetAllNodes()) {
        node_to_index_[node_id] = index++;
    }

    int num_nodes = mesh_->GetNumNodes();
    std::cout << "Number of nodes: " << num_nodes << std::endl;

    // Initialize DOF arrays
    displacements_.resize(num_nodes, {0.0, 0.0, 0.0});
    velocities_.resize(num_nodes, {0.0, 0.0, 0.0});
    accelerations_.resize(num_nodes, {0.0, 0.0, 0.0});
    masses_.resize(num_nodes, 0.0);
    internal_forces_.resize(num_nodes, {0.0, 0.0, 0.0});
    external_forces_.resize(num_nodes, {0.0, 0.0, 0.0});

    // Create elements
    CreateElements();
    std::cout << "Number of elements: " << elements_.size() << std::endl;

    // Compute mass matrix
    ComputeMassMatrix();

    // Determine time step if auto mode
    if (params_.auto_time_step) {
        double dt_crit = ComputeCriticalTimeStep();
        params_.time_step = params_.time_step_scale * dt_crit;
        std::cout << "Critical time step: " << dt_crit << " s" << std::endl;
        std::cout << "Using time step: " << params_.time_step << " s" << std::endl;
    }
    else {
        std::cout << "Using user-specified time step: " << params_.time_step << " s" << std::endl;
    }

    current_step_ = 0;
    current_time_ = 0.0;
    strain_energy_ = 0.0;

    std::cout << "Initialization complete." << std::endl;
    std::cout << "=====================================" << std::endl;
}

void ExplicitSolver::CreateElements() {
    // Determine which material to use for each element
    std::shared_ptr<Material> default_material = nullptr;
    if (materials_.count("ALL")) {
        default_material = materials_["ALL"];
    }

    for (const auto& [elem_id, elem_conn] : mesh_->GetAllElements()) {
        // Determine material
        std::shared_ptr<Material> material = default_material;

        // Check if element belongs to any element set with assigned material
        for (const auto& [set_name, mat] : materials_) {
            if (set_name == "ALL") continue;
            const auto* eset = mesh_->GetElementSet(set_name);
            if (eset) {
                auto it = std::find(eset->elements.begin(), eset->elements.end(), elem_id);
                if (it != eset->elements.end()) {
                    material = mat;
                    break;
                }
            }
        }

        if (!material) {
            throw std::runtime_error("No material assigned to element " + std::to_string(elem_id));
        }

        // Create element based on type
        std::shared_ptr<Element> element;
        if (elem_conn.type == "C3D8" || elem_conn.type == "C3D8R") {
            element = std::make_shared<HexElement>(elem_id, elem_conn.nodes, material);
        }
        else if (elem_conn.type == "C3D4") {
            element = std::make_shared<TetElement>(elem_id, elem_conn.nodes, material);
        }
        else {
            std::cerr << "Warning: Unsupported element type " << elem_conn.type
                      << " for element " << elem_id << ", skipping" << std::endl;
            continue;
        }

        elements_.push_back(element);
    }
}

void ExplicitSolver::ComputeMassMatrix() {
    // Lumped mass matrix
    for (const auto& element : elements_) {
        const auto& node_ids = element->GetNodeIds();
        int num_nodes = element->GetNumNodes();

        // Get node coordinates
        std::vector<std::array<double, 3>> coords(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            const auto& node = mesh_->GetNode(node_ids[i]);
            coords[i] = node.coords;
        }

        // Compute element mass and distribute to nodes
        std::vector<double> lumped_mass(num_nodes);
        element->ComputeMassMatrix(coords, lumped_mass);

        // Add to global mass vector
        for (int i = 0; i < num_nodes; ++i) {
            int global_idx = node_to_index_[node_ids[i]];
            masses_[global_idx] += lumped_mass[i];
        }
    }

    // Check for zero masses
    for (size_t i = 0; i < masses_.size(); ++i) {
        if (masses_[i] < 1e-12) {
            std::cerr << "Warning: Node " << i << " has very small mass" << std::endl;
            masses_[i] = 1e-12;  // Prevent division by zero
        }
    }
}

double ExplicitSolver::ComputeCriticalTimeStep() {
    double min_dt = std::numeric_limits<double>::max();

    for (const auto& element : elements_) {
        const auto& node_ids = element->GetNodeIds();
        int num_nodes = element->GetNumNodes();

        // Get node coordinates
        std::vector<std::array<double, 3>> coords(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            const auto& node = mesh_->GetNode(node_ids[i]);
            coords[i] = node.coords;
        }

        double dt_elem = element->ComputeCriticalTimeStep(coords);
        min_dt = std::min(min_dt, dt_elem);
    }

    return min_dt;
}

void ExplicitSolver::Solve() {
    std::cout << "\n=== Starting Explicit Dynamics Simulation ===" << std::endl;
    std::cout << "Total steps: " << params_.num_steps << std::endl;
    std::cout << "Time step: " << params_.time_step << " s" << std::endl;
    std::cout << "Total time: " << params_.num_steps * params_.time_step << " s" << std::endl;
    std::cout << "============================================\n" << std::endl;

    for (current_step_ = 0; current_step_ < params_.num_steps; ++current_step_) {
        Step();

        // Progress output
        if (current_step_ % params_.output_interval == 0) {
            double progress = 100.0 * current_step_ / params_.num_steps;
            std::cout << "Step " << current_step_ << " / " << params_.num_steps
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                      << " Time: " << std::scientific << current_time_ << " s" << std::endl;

            if (output_callback_) {
                output_callback_(current_step_, current_time_);
            }
        }
    }

    std::cout << "\n=== Simulation Complete ===" << std::endl;
}

void ExplicitSolver::Step() {
    // 1. Compute internal forces
    ComputeInternalForces();

    // 2. Apply external loads
    ApplyExternalLoads();

    // 3. Update velocities and displacements
    UpdateVelocitiesAndDisplacements();

    // 4. Apply boundary conditions
    ApplyBoundaryConditions();

    // 5. Update time
    current_time_ += params_.time_step;
}

void ExplicitSolver::ComputeInternalForces() {
    // Zero out internal forces and recompute strain energy
    std::fill(internal_forces_.begin(), internal_forces_.end(), std::array<double, 3>{0.0, 0.0, 0.0});
    strain_energy_ = 0.0;

    // Loop over all elements
    for (const auto& element : elements_) {
        const auto& node_ids = element->GetNodeIds();
        int num_nodes = element->GetNumNodes();

        // Get node coordinates and displacements
        std::vector<std::array<double, 3>> coords(num_nodes);
        std::vector<std::array<double, 3>> disps(num_nodes);

        for (int i = 0; i < num_nodes; ++i) {
            const auto& node = mesh_->GetNode(node_ids[i]);
            int global_idx = node_to_index_[node_ids[i]];
            coords[i] = node.coords;
            disps[i] = displacements_[global_idx];
        }

        // Compute element internal forces
        std::vector<std::array<double, 3>> elem_forces(num_nodes);
        element->ComputeInternalForce(coords, disps, elem_forces);

        // Update strain energy contribution
        double volume = element->ComputeVolume(coords);
        const auto& stress = element->GetStress();
        const auto& strain = element->GetStrain();
        double work = 0.0;
        for (int i = 0; i < 6; ++i) {
            work += stress[i] * strain[i];
        }
        strain_energy_ += 0.5 * work * volume;

        // Assemble to global force vector
        for (int i = 0; i < num_nodes; ++i) {
            int global_idx = node_to_index_[node_ids[i]];
            internal_forces_[global_idx][0] += elem_forces[i][0];
            internal_forces_[global_idx][1] += elem_forces[i][1];
            internal_forces_[global_idx][2] += elem_forces[i][2];
        }
    }
}

void ExplicitSolver::ApplyExternalLoads() {
    // Zero out external forces
    std::fill(external_forces_.begin(), external_forces_.end(), std::array<double, 3>{0.0, 0.0, 0.0});

    for (const auto& load : loads_) {
        double value = load.value;

        // Apply time function if specified
        if (load.time_function) {
            value *= load.time_function(current_time_);
        }

        if (load.type == LoadType::FORCE) {
            for (int node_id : load.nodes) {
                int idx = node_to_index_[node_id];
                external_forces_[idx][load.component] += value;
            }
        }
        // TODO: Implement other load types (pressure, body force, gravity)
    }
}

void ExplicitSolver::UpdateVelocitiesAndDisplacements() {
    // Central difference time integration
    const double dt = params_.time_step;
    const double damping = params_.damping;

    for (size_t i = 0; i < displacements_.size(); ++i) {
        for (int comp = 0; comp < 3; ++comp) {
            // Total force
            double f_total = external_forces_[i][comp] - internal_forces_[i][comp];

            // Damping force
            if (damping > 0.0) {
                f_total -= damping * velocities_[i][comp];
            }

            // Acceleration: a = F / m
            accelerations_[i][comp] = f_total / masses_[i];

            // Update velocity: v^(n+1/2) = v^(n-1/2) + a^n * dt
            velocities_[i][comp] += accelerations_[i][comp] * dt;

            // Update displacement: u^(n+1) = u^n + v^(n+1/2) * dt
            displacements_[i][comp] += velocities_[i][comp] * dt;
        }
    }
}

void ExplicitSolver::ApplyBoundaryConditions() {
    for (const auto& bc : boundary_conditions_) {
        if (bc.type == BCType::FIXED) {
            for (int node_id : bc.nodes) {
                int idx = node_to_index_[node_id];

                if (bc.component == -1) {
                    // Fix all components
                    displacements_[idx] = {0.0, 0.0, 0.0};
                    velocities_[idx] = {0.0, 0.0, 0.0};
                    accelerations_[idx] = {0.0, 0.0, 0.0};
                }
                else {
                    // Fix specific component
                    displacements_[idx][bc.component] = 0.0;
                    velocities_[idx][bc.component] = 0.0;
                    accelerations_[idx][bc.component] = 0.0;
                }
            }
        }
        // TODO: Implement other BC types
    }
}

std::array<double, 3> ExplicitSolver::GetNodeDisplacement(int node_id) const {
    int idx = node_to_index_.at(node_id);
    return displacements_[idx];
}

std::array<double, 3> ExplicitSolver::GetNodeVelocity(int node_id) const {
    int idx = node_to_index_.at(node_id);
    return velocities_[idx];
}

std::vector<ExplicitSolver::NodeResult> ExplicitSolver::GetNodeResults() const {
    std::vector<NodeResult> results;
    results.reserve(mesh_->GetNumNodes());

    for (const auto& [node_id, node] : mesh_->GetAllNodes()) {
        NodeResult result;
        result.id = node_id;
        result.coords = node.coords;
        auto it = node_to_index_.find(node_id);
        if (it != node_to_index_.end()) {
            result.displacement = displacements_[it->second];
        } else {
            result.displacement = {0.0, 0.0, 0.0};
        }
        results.push_back(result);
    }

    std::sort(results.begin(), results.end(),
              [](const NodeResult& a, const NodeResult& b) { return a.id < b.id; });

    return results;
}

std::vector<ExplicitSolver::ElementResult> ExplicitSolver::GetElementResults() const {
    std::vector<ElementResult> results;
    results.reserve(elements_.size());

    for (const auto& element : elements_) {
        ElementResult result;
        result.id = element->GetId();
        result.type = element->GetType();

        const auto& node_ids = element->GetNodeIds();
        int num_nodes = element->GetNumNodes();
        std::vector<std::array<double, 3>> coords(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            coords[i] = mesh_->GetNode(node_ids[i]).coords;
        }

        result.volume = element->ComputeVolume(coords);
        result.stress = element->GetStress();
        result.strain = element->GetStrain();

        results.push_back(result);
    }

    std::sort(results.begin(), results.end(),
              [](const ElementResult& a, const ElementResult& b) { return a.id < b.id; });

    return results;
}

double ExplicitSolver::GetKineticEnergy() const {
    double ke = 0.0;
    for (size_t i = 0; i < velocities_.size(); ++i) {
        double v_mag_sq = velocities_[i][0] * velocities_[i][0] +
                          velocities_[i][1] * velocities_[i][1] +
                          velocities_[i][2] * velocities_[i][2];
        ke += 0.5 * masses_[i] * v_mag_sq;
    }
    return ke;
}

double ExplicitSolver::GetStrainEnergy() const {
    return strain_energy_;
}

double ExplicitSolver::GetTotalEnergy() const {
    return GetKineticEnergy() + GetStrainEnergy();
}

void ExplicitSolver::SetOutputCallback(std::function<void(int, double)> callback) {
    output_callback_ = callback;
}

void ExplicitSolver::WriteResults(const std::string& filename) {
    // Simple CSV output for now
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open results file: " + filename);
    }

    file << "NodeID,X,Y,Z,Ux,Uy,Uz\n";

    for (const auto& [node_id, node] : mesh_->GetAllNodes()) {
        int idx = node_to_index_[node_id];
        file << node_id << ","
             << node.coords[0] << "," << node.coords[1] << "," << node.coords[2] << ","
             << displacements_[idx][0] << "," << displacements_[idx][1] << "," << displacements_[idx][2] << "\n";
    }

    file.close();
    std::cout << "Results written to " << filename << std::endl;

    // Derive element results file name
    std::string element_filename = filename;
    auto dot_pos = element_filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        element_filename.insert(dot_pos, "_elements");
    } else {
        element_filename += "_elements.csv";
    }

    std::ofstream elem_file(element_filename);
    if (!elem_file) {
        throw std::runtime_error("Failed to open element results file: " + element_filename);
    }

    elem_file << "ElementID,Type,Volume,"
              << "Sxx,Syy,Szz,Sxy,Sxz,Syz,"
              << "Exx,Eyy,Ezz,Exy,Exz,Eyz\n";

    std::vector<std::shared_ptr<Element>> sorted_elements = elements_;
    std::sort(sorted_elements.begin(), sorted_elements.end(),
              [](const std::shared_ptr<Element>& a, const std::shared_ptr<Element>& b) {
                  return a->GetId() < b->GetId();
              });

    for (const auto& element : sorted_elements) {
        const auto& node_ids = element->GetNodeIds();
        int num_nodes = element->GetNumNodes();
        std::vector<std::array<double, 3>> coords(num_nodes);
        for (int i = 0; i < num_nodes; ++i) {
            coords[i] = mesh_->GetNode(node_ids[i]).coords;
        }
        double volume = element->ComputeVolume(coords);
        const auto& stress = element->GetStress();
        const auto& strain = element->GetStrain();

        elem_file << element->GetId() << "," << element->GetType() << "," << volume;
        for (int i = 0; i < 6; ++i) {
            elem_file << "," << stress[i];
        }
        for (int i = 0; i < 6; ++i) {
            elem_file << "," << strain[i];
        }
        elem_file << "\n";
    }

    elem_file.close();
    std::cout << "Element results written to " << element_filename << std::endl;
}

} // namespace femml
