#include "material/LinearElastic.hpp"
#include <stdexcept>

namespace femml {

LinearElastic::LinearElastic(const std::string& name, double E, double nu, double rho)
    : Material(name), E_(E), nu_(nu) {
    if (E <= 0.0) {
        throw std::invalid_argument("Young's modulus must be positive");
    }
    if (nu < -1.0 || nu >= 0.5) {
        throw std::invalid_argument("Poisson's ratio must be in range [-1, 0.5)");
    }
    density_ = rho;
    ComputeLameParameters();
}

void LinearElastic::ComputeLameParameters() {
    mu_ = E_ / (2.0 * (1.0 + nu_));
    lambda_ = E_ * nu_ / ((1.0 + nu_) * (1.0 - 2.0 * nu_));
}

std::array<double, 6> LinearElastic::ComputeStressIncrement(
    const std::array<double, 6>& strain_increment,
    const std::array<double, 6>& /*current_strain*/,
    const std::array<double, 6>& /*current_stress*/) const {

    // Convert engineering shear increments to tensor form
    double deps_xx = strain_increment[0];
    double deps_yy = strain_increment[1];
    double deps_zz = strain_increment[2];
    double deps_xy = strain_increment[3] / 2.0;
    double deps_xz = strain_increment[4] / 2.0;
    double deps_yz = strain_increment[5] / 2.0;

    double trace_inc = deps_xx + deps_yy + deps_zz;

    std::array<double, 6> stress_inc{};

    // Normal stress increments
    stress_inc[0] = lambda_ * trace_inc + 2.0 * mu_ * deps_xx;
    stress_inc[1] = lambda_ * trace_inc + 2.0 * mu_ * deps_yy;
    stress_inc[2] = lambda_ * trace_inc + 2.0 * mu_ * deps_zz;

    // Shear stress increments
    stress_inc[3] = 2.0 * mu_ * deps_xy;
    stress_inc[4] = 2.0 * mu_ * deps_xz;
    stress_inc[5] = 2.0 * mu_ * deps_yz;

    return stress_inc;
}

std::array<std::array<double, 6>, 6> LinearElastic::ComputeTangent(
    const std::array<double, 6>& strain) const {

    std::array<std::array<double, 6>, 6> C{};

    // Diagonal terms for normal stresses
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i][j] = (i == j) ? (lambda_ + 2.0 * mu_) : lambda_;
        }
    }

    // Diagonal terms for shear stresses
    for (int i = 3; i < 6; ++i) {
        C[i][i] = mu_;
    }

    return C;
}

std::shared_ptr<Material> LinearElastic::Clone() const {
    return std::make_shared<LinearElastic>(name_, E_, nu_, density_);
}

// Helper functions for common materials
std::shared_ptr<LinearElastic> CreateAluminum() {
    return std::make_shared<LinearElastic>("Aluminum", 70e9, 0.33, 2700.0);
}

std::shared_ptr<LinearElastic> CreateSteel() {
    return std::make_shared<LinearElastic>("Steel", 210e9, 0.3, 7800.0);
}

std::shared_ptr<LinearElastic> CreateTitanium() {
    return std::make_shared<LinearElastic>("Titanium", 110e9, 0.34, 4500.0);
}

} // namespace femml
