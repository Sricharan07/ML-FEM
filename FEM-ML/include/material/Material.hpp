#pragma once

#include <array>
#include <string>
#include <memory>
#include <map>

namespace femml {

// Material base class
class Material {
public:
    Material(const std::string& name) : name_(name), density_(0.0) {}
    virtual ~Material() = default;

    // Pure virtual: return stress increment for a strain increment (explicit style)
    // Inputs represent engineering Voigt components from the previous state.
    // current_strain/current_stress correspond to the stored totals at time n.
    virtual std::array<double, 6> ComputeStressIncrement(
        const std::array<double, 6>& strain_increment,
        const std::array<double, 6>& current_strain,
        const std::array<double, 6>& current_stress) const = 0;

    // Optional: compute tangent stiffness matrix (for implicit solvers)
    virtual std::array<std::array<double, 6>, 6> ComputeTangent(
        const std::array<double, 6>& strain) const {
        std::array<std::array<double, 6>, 6> tangent{};
        for (auto& row : tangent) {
            row.fill(0.0);
        }
        return tangent;
    }

    // Material properties
    const std::string& GetName() const { return name_; }
    double GetDensity() const { return density_; }
    void SetDensity(double rho) { density_ = rho; }

    // Material type identifier
    virtual std::string GetType() const = 0;

    // Clone for creating copies
    virtual std::shared_ptr<Material> Clone() const = 0;

protected:
    std::string name_;
    double density_;
};

// Material library
class MaterialLibrary {
public:
    static MaterialLibrary& Instance();

    void AddMaterial(const std::shared_ptr<Material>& material);
    std::shared_ptr<Material> GetMaterial(const std::string& name) const;
    void Clear();

    void PrintSummary() const;

private:
    MaterialLibrary() = default;
    std::map<std::string, std::shared_ptr<Material>> materials_;
};

} // namespace femml
