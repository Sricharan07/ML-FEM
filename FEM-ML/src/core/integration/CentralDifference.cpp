#include "integration/CentralDifference.hpp"

namespace femml {

CentralDifference::CentralDifference(double dt, double damping)
    : TimeIntegrator(dt), damping_(damping) {
}

void CentralDifference::Update(
    const std::vector<double>& masses,
    const std::vector<std::array<double, 3>>& forces,
    std::vector<std::array<double, 3>>& displacements,
    std::vector<std::array<double, 3>>& velocities,
    std::vector<std::array<double, 3>>& accelerations) {

    const size_t num_nodes = masses.size();

    // Update each node
    for (size_t i = 0; i < num_nodes; ++i) {
        for (int comp = 0; comp < 3; ++comp) {
            // Apply damping to force
            double f_damped = forces[i][comp] - damping_ * velocities[i][comp];

            // Compute acceleration: a = F / m
            accelerations[i][comp] = f_damped / masses[i];

            // Update velocity: v^(n+1/2) = v^(n-1/2) + a^n * dt
            velocities[i][comp] += accelerations[i][comp] * dt_;

            // Update displacement: u^(n+1) = u^n + v^(n+1/2) * dt
            displacements[i][comp] += velocities[i][comp] * dt_;
        }
    }
}

} // namespace femml
