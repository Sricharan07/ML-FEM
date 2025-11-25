#pragma once

#include "element/Element.hpp"

namespace femml {

// 8-node hexahedral element (C3D8)
class HexElement : public Element {
public:
    HexElement(int id, const std::vector<int>& node_ids,
               const std::shared_ptr<Material>& material);
    ~HexElement() override = default;

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

    std::string GetType() const override { return "C3D8"; }
    int GetNumNodes() const override { return 8; }

private:
    // Shape function derivatives at element center
    void ComputeShapeGradients(
        const std::vector<std::array<double, 3>>& node_coords,
        std::array<std::array<double, 3>, 8>& shape_grads,
        double& det_jacobian);

    // B matrix (strain-displacement)
    void BuildBMatrix(
        const std::array<std::array<double, 3>, 8>& shape_grads,
        std::array<std::array<double, 24>, 6>& B);

    // Element characteristic length
    double ComputeCharacteristicLength(
        const std::vector<std::array<double, 3>>& node_coords);
};

// 8-node hexahedral element with reduced integration (C3D8R)
class HexElementReduced : public HexElement {
public:
    using HexElement::HexElement;

    std::string GetType() const override { return "C3D8R"; }

    // Uses single gauss point at center (already implemented in base)
};

} // namespace femml
