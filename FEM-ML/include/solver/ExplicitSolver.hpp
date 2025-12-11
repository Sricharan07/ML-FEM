#pragma once

#include "mesh/Mesh.hpp"
#include "element/Element.hpp"
#include "material/Material.hpp"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <functional>

namespace femml {

// Boundary condition types
enum class BCType {
    FIXED,
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION
};

// Load types
enum class LoadType {
    FORCE,
    PRESSURE,
    BODY_FORCE,
    GRAVITY
};

// Boundary condition
struct BoundaryCondition {
    std::vector<int> nodes;
    BCType type;
    int component;  // 0=x, 1=y, 2=z, -1=all
    double value;
    double ramp_time; // Time to ramp to target displacement (for DISPLACEMENT BCs)

    BoundaryCondition()
        : type(BCType::FIXED),
          component(-1),
          value(0.0),
          ramp_time(0.0) {}
};

// Load definition
struct Load {
    LoadType type;
    std::vector<int> nodes;  // For nodal loads
    std::string surface;     // For pressure loads
    int component;  // Direction for forces
    double value;
    std::function<double(double)> time_function;  // Optional time-dependent load
};

// Solver parameters
struct SolverParams {
    double time_step;
    int num_steps;
    double damping;
    int output_interval;
    bool auto_time_step;
    double time_step_scale;  // Safety factor for critical time step

    SolverParams()
        : time_step(1e-7), num_steps(1000), damping(0.0),
          output_interval(10), auto_time_step(false), time_step_scale(0.9) {}
};

// Explicit dynamics solver
class ExplicitSolver {
public:
    struct NodeResult {
        int id;
        std::array<double, 3> coords;
        std::array<double, 3> displacement;
    };

    struct ElementResult {
        int id;
        std::string type;
        double volume;
        std::array<double, 6> stress;
        std::array<double, 6> strain;
    };

    ExplicitSolver(const std::shared_ptr<Mesh>& mesh);
    virtual ~ExplicitSolver() = default;

    // Setup
    void SetMaterial(const std::string& element_set, const std::shared_ptr<Material>& material);
    void SetMaterial(const std::shared_ptr<Material>& material);  // For all elements
    void AddBoundaryCondition(const BoundaryCondition& bc);
    void AddLoad(const Load& load);
    void SetParameters(const SolverParams& params);

    // Initialize solver
    virtual void Initialize();

    // Solve
    virtual void Solve();

    // Step-by-step solving (for GUI updates)
    virtual void Step();
    bool IsFinished() const { return current_step_ >= params_.num_steps; }
    int GetCurrentStep() const { return current_step_; }
    double GetCurrentTime() const { return current_time_; }
    int GetTotalSteps() const { return params_.num_steps; }
    double GetTimeStepSize() const { return params_.time_step; }

    // Results
    const std::vector<std::array<double, 3>>& GetDisplacements() const { return displacements_; }
    const std::vector<std::array<double, 3>>& GetVelocities() const { return velocities_; }
    const std::vector<std::array<double, 3>>& GetAccelerations() const { return accelerations_; }

    std::array<double, 3> GetNodeDisplacement(int node_id) const;
    std::array<double, 3> GetNodeVelocity(int node_id) const;

    std::vector<NodeResult> GetNodeResults() const;
    std::vector<ElementResult> GetElementResults() const;

    // Element results
    const std::vector<std::shared_ptr<Element>>& GetElements() const { return elements_; }

    // Energy
    double GetKineticEnergy() const;
    double GetStrainEnergy() const;
    double GetTotalEnergy() const;

    // Output
    void SetOutputCallback(std::function<void(int, double)> callback);
    void WriteResults(const std::string& filename);

protected:
    void CreateElements();
    void ComputeMassMatrix();
    double ComputeCriticalTimeStep();
    void ApplyBoundaryConditions();
    void ComputeInternalForces();
    void ApplyExternalLoads();
    void UpdateVelocitiesAndDisplacements();

    std::shared_ptr<Mesh> mesh_;
    std::vector<std::shared_ptr<Element>> elements_;
    std::map<std::string, std::shared_ptr<Material>> materials_;

    // DOF arrays (indexed by node_id)
    std::map<int, int> node_to_index_;  // Map node ID to array index
    std::vector<std::array<double, 3>> displacements_;
    std::vector<std::array<double, 3>> velocities_;
    std::vector<std::array<double, 3>> accelerations_;
    std::vector<double> masses_;  // Lumped mass per node
    std::vector<std::array<double, 3>> internal_forces_;
    std::vector<std::array<double, 3>> external_forces_;

    // Boundary conditions and loads
    std::vector<BoundaryCondition> boundary_conditions_;
    std::vector<Load> loads_;

    // Solver parameters
    SolverParams params_;
    int current_step_;
    double current_time_;

    // Output callback
    std::function<void(int, double)> output_callback_;

    double strain_energy_;
};

} // namespace femml
