#pragma once

#include "integration/TimeIntegrator.hpp"

namespace femml {

// Central difference explicit time integrator
class CentralDifference : public TimeIntegrator {
public:
    CentralDifference(double dt, double damping = 0.0);
    ~CentralDifference() override = default;

    void Update(
        const std::vector<double>& masses,
        const std::vector<std::array<double, 3>>& forces,
        std::vector<std::array<double, 3>>& displacements,
        std::vector<std::array<double, 3>>& velocities,
        std::vector<std::array<double, 3>>& accelerations) override;

    void SetDamping(double damping) { damping_ = damping; }
    double GetDamping() const { return damping_; }

private:
    double damping_;
};

} // namespace femml
