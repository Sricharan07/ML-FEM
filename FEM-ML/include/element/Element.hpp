#pragma once

#include "material/Material.hpp"
#include <array>
#include <vector>
#include <memory>
#include <map>

namespace femml {

// Base element class
class Element {
public:
    Element(int id, const std::vector<int>& node_ids,
            const std::shared_ptr<Material>& material);
    virtual ~Element() = default;

    // Pure virtual: compute element internal force
    virtual void ComputeInternalForce(
        const std::vector<std::array<double, 3>>& node_coords,
        const std::vector<std::array<double, 3>>& node_displacements,
        std::vector<std::array<double, 3>>& node_forces) = 0;

    // Optional: compute element mass matrix (for consistent mass)
    virtual void ComputeMassMatrix(
        const std::vector<std::array<double, 3>>& node_coords,
        std::vector<double>& lumped_mass) = 0;

    // Get element volume
    virtual double ComputeVolume(
        const std::vector<std::array<double, 3>>& node_coords) = 0;

    // Get critical time step (for explicit dynamics)
    virtual double ComputeCriticalTimeStep(
        const std::vector<std::array<double, 3>>& node_coords) = 0;

    // Accessors
    int GetId() const { return id_; }
    const std::vector<int>& GetNodeIds() const { return node_ids_; }
    const std::shared_ptr<Material>& GetMaterial() const { return material_; }
    void SetMaterial(const std::shared_ptr<Material>& material) { material_ = material; }

    // Element type
    virtual std::string GetType() const = 0;
    virtual int GetNumNodes() const = 0;

    // State variables (for history-dependent materials)
    void SetStateVariable(const std::string& name, double value);
    double GetStateVariable(const std::string& name) const;

    // Get element strains and stresses (for output)
    const std::array<double, 6>& GetStrain() const { return strain_; }
    const std::array<double, 6>& GetStress() const { return stress_; }

protected:
    int id_;
    std::vector<int> node_ids_;
    std::shared_ptr<Material> material_;

    // Element state
    std::array<double, 6> strain_;
    std::array<double, 6> stress_;
    std::map<std::string, double> state_variables_;
};

inline Element::Element(int id,
                        const std::vector<int>& node_ids,
                        const std::shared_ptr<Material>& material)
    : id_(id),
      node_ids_(node_ids),
      material_(material),
      strain_{},
      stress_{} {}

inline void Element::SetStateVariable(const std::string& name, double value) {
    state_variables_[name] = value;
}

inline double Element::GetStateVariable(const std::string& name) const {
    auto it = state_variables_.find(name);
    return (it != state_variables_.end()) ? it->second : 0.0;
}

} // namespace femml
