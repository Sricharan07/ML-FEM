#pragma once

#include "element/Element.hpp"

namespace femml {

// 4-node tetrahedral element (C3D4)
class TetElement : public Element {
public:
    TetElement(int id, const std::vector<int>& node_ids,
               const std::shared_ptr<Material>& material);
    ~TetElement() override = default;

    void ComputeInternalForce(
        const std::vector<std::array<double, 3>>& node_coords,
        const std::vector<std::array<double, 3>>& node_displacements,
        std::vector<std::array<double, 3>>& node_forces) override;

    void ComputeMassMatrix(
        const std::vector<std::array<double, 3>>& node_coords,
        std::vector<double>& lumped_mass) override;

    double ComputeVolume(
        const std::vector<std::array<double, 3>>& node_coords) override;

    double ComputeCriticalTimeStep(
        const std::vector<std::array<double, 3>>& node_coords) override;

    std::string GetType() const override { return "C3D4"; }
    int GetNumNodes() const override { return 4; }

private:
    // Compute shape function gradients (constant for linear tet)
    void ComputeShapeGradients(
        const std::vector<std::array<double, 3>>& node_coords,
        std::array<std::array<double, 3>, 4>& shape_grads,
        double& volume);

    // B matrix (strain-displacement)
    void BuildBMatrix(
        const std::array<std::array<double, 3>, 4>& shape_grads,
        std::array<std::array<double, 12>, 6>& B);
};

} // namespace femml
