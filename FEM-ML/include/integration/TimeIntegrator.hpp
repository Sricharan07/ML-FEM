#pragma once

#include <array>
#include <vector>

namespace femml {

// Base class for time integrators
class TimeIntegrator {
public:
    TimeIntegrator(double dt) : dt_(dt) {}
    virtual ~TimeIntegrator() = default;

    // Update state: positions, velocities, accelerations
    virtual void Update(
        const std::vector<double>& masses,
        const std::vector<std::array<double, 3>>& forces,
        std::vector<std::array<double, 3>>& displacements,
        std::vector<std::array<double, 3>>& velocities,
        std::vector<std::array<double, 3>>& accelerations) = 0;

    double GetTimeStep() const { return dt_; }
    void SetTimeStep(double dt) { dt_ = dt; }

protected:
    double dt_;
};

} // namespace femml
