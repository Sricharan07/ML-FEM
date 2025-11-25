#pragma once

#include "material/Material.hpp"

namespace femml {

class LinearElastic : public Material {
public:
    LinearElastic(const std::string& name, double E, double nu, double rho);
    ~LinearElastic() override = default;

    std::array<double, 6> ComputeStressIncrement(
        const std::array<double, 6>& strain_increment,
        const std::array<double, 6>& current_strain,
        const std::array<double, 6>& current_stress) const override;

    std::array<std::array<double, 6>, 6> ComputeTangent(
        const std::array<double, 6>& strain) const override;

    std::string GetType() const override { return "LinearElastic"; }

    std::shared_ptr<Material> Clone() const override;

    // Accessors
    double GetYoungsModulus() const { return E_; }
    double GetPoissonsRatio() const { return nu_; }
    double GetLameFirst() const { return lambda_; }
    double GetLameSecond() const { return mu_; }

private:
    void ComputeLameParameters();

    double E_;       // Young's modulus
    double nu_;      // Poisson's ratio
    double lambda_;  // Lamé first parameter
    double mu_;      // Lamé second parameter (shear modulus)
};

// Helper function to create common materials
std::shared_ptr<LinearElastic> CreateAluminum();
std::shared_ptr<LinearElastic> CreateSteel();
std::shared_ptr<LinearElastic> CreateTitanium();

} // namespace femml
