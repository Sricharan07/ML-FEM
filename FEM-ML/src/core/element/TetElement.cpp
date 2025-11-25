#include "element/TetElement.hpp"
#include <cmath>
#include <stdexcept>

namespace femml {

TetElement::TetElement(int id, const std::vector<int>& node_ids,
                       const std::shared_ptr<Material>& material)
    : Element(id, node_ids, material) {
    if (node_ids.size() != 4) {
        throw std::invalid_argument("Tet element must have 4 nodes");
    }
}

void TetElement::ComputeInternalForce(
    const std::vector<std::array<double, 3>>& node_coords,
    const std::vector<std::array<double, 3>>& node_displacements,
    std::vector<std::array<double, 3>>& node_forces) {

    if (node_coords.size() != 4 || node_displacements.size() != 4 || node_forces.size() != 4) {
        throw std::invalid_argument("TetElement: incorrect array sizes");
    }

    // Compute shape function gradients and volume
    std::array<std::array<double, 3>, 4> shape_grads;
    double volume;
    ComputeShapeGradients(node_coords, shape_grads, volume);

    // Build B matrix (6x12)
    std::array<std::array<double, 12>, 6> B;
    BuildBMatrix(shape_grads, B);

    // Flatten displacements to vector [u1x, u1y, u1z, u2x, ...]
    std::array<double, 12> u_flat{};
    for (int i = 0; i < 4; ++i) {
        u_flat[i * 3 + 0] = node_displacements[i][0];
        u_flat[i * 3 + 1] = node_displacements[i][1];
        u_flat[i * 3 + 2] = node_displacements[i][2];
    }

    // Compute strain: ε = B * u
    std::array<double, 6> new_strain{};
    for (int i = 0; i < 6; ++i) {
        double val = 0.0;
        for (int j = 0; j < 12; ++j) {
            val += B[i][j] * u_flat[j];
        }
        new_strain[i] = val;
    }

    std::array<double, 6> strain_increment{};
    for (int i = 0; i < 6; ++i) {
        strain_increment[i] = new_strain[i] - strain_[i];
    }

    const auto stress_increment =
        material_->ComputeStressIncrement(strain_increment, strain_, stress_);

    for (int i = 0; i < 6; ++i) {
        stress_[i] += stress_increment[i];
    }
    strain_ = new_strain;

    // Compute internal force: f_int = B^T * σ * volume
    std::array<double, 12> f_int_flat{};
    for (int j = 0; j < 12; ++j) {
        double val = 0.0;
        for (int i = 0; i < 6; ++i) {
            val += B[i][j] * stress_[i];
        }
        f_int_flat[j] = val * volume;
    }

    // Unflatten to node forces
    for (int i = 0; i < 4; ++i) {
        node_forces[i][0] = f_int_flat[i * 3 + 0];
        node_forces[i][1] = f_int_flat[i * 3 + 1];
        node_forces[i][2] = f_int_flat[i * 3 + 2];
    }
}

void TetElement::ComputeMassMatrix(
    const std::vector<std::array<double, 3>>& node_coords,
    std::vector<double>& lumped_mass) {

    if (lumped_mass.size() != 4) {
        lumped_mass.resize(4);
    }

    double volume = ComputeVolume(node_coords);
    double total_mass = material_->GetDensity() * volume;
    double mass_per_node = total_mass / 4.0;

    for (int i = 0; i < 4; ++i) {
        lumped_mass[i] = mass_per_node;
    }
}

double TetElement::ComputeVolume(
    const std::vector<std::array<double, 3>>& node_coords) {

    std::array<std::array<double, 3>, 4> shape_grads;
    double volume;
    ComputeShapeGradients(node_coords, shape_grads, volume);
    return volume;
}

double TetElement::ComputeCriticalTimeStep(
    const std::vector<std::array<double, 3>>& node_coords) {

    // Characteristic length: use minimum edge length
    double min_length = std::numeric_limits<double>::max();

    const int edges[6][2] = {
        {0, 1}, {1, 2}, {2, 0},  // Base triangle
        {0, 3}, {1, 3}, {2, 3}   // Edges to apex
    };

    for (int e = 0; e < 6; ++e) {
        int n1 = edges[e][0];
        int n2 = edges[e][1];

        double dx = node_coords[n2][0] - node_coords[n1][0];
        double dy = node_coords[n2][1] - node_coords[n1][1];
        double dz = node_coords[n2][2] - node_coords[n1][2];

        double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        min_length = std::min(min_length, length);
    }

    double rho = material_->GetDensity();
    double E = 70e9;  // TODO: Get from material
    double wave_speed = std::sqrt(E / rho);

    return min_length / wave_speed;
}

void TetElement::ComputeShapeGradients(
    const std::vector<std::array<double, 3>>& node_coords,
    std::array<std::array<double, 3>, 4>& shape_grads,
    double& volume) {

    // For linear tetrahedron, shape function gradients are constant
    // N1 = 1 - ξ - η - ζ
    // N2 = ξ
    // N3 = η
    // N4 = ζ

    // Construct matrix with node coordinates
    // [x1-x4  y1-y4  z1-z4]
    // [x2-x4  y2-y4  z2-z4]
    // [x3-x4  y3-y4  z3-z4]

    std::array<std::array<double, 3>, 3> J;
    for (int i = 0; i < 3; ++i) {
        J[i][0] = node_coords[i][0] - node_coords[3][0];
        J[i][1] = node_coords[i][1] - node_coords[3][1];
        J[i][2] = node_coords[i][2] - node_coords[3][2];
    }

    // Compute determinant (6 * volume)
    double det = J[0][0] * (J[1][1] * J[2][2] - J[2][1] * J[1][2]) -
                 J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2]) +
                 J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]);

    volume = std::abs(det) / 6.0;

    if (std::abs(det) < 1e-12) {
        throw std::runtime_error("Tet element has zero or negative volume");
    }

    // Compute inverse Jacobian
    double inv_det = 1.0 / det;
    std::array<std::array<double, 3>, 3> invJ;
    invJ[0][0] =  (J[1][1] * J[2][2] - J[2][1] * J[1][2]) * inv_det;
    invJ[0][1] = -(J[0][1] * J[2][2] - J[2][1] * J[0][2]) * inv_det;
    invJ[0][2] =  (J[0][1] * J[1][2] - J[1][1] * J[0][2]) * inv_det;
    invJ[1][0] = -(J[1][0] * J[2][2] - J[2][0] * J[1][2]) * inv_det;
    invJ[1][1] =  (J[0][0] * J[2][2] - J[2][0] * J[0][2]) * inv_det;
    invJ[1][2] = -(J[0][0] * J[1][2] - J[1][0] * J[0][2]) * inv_det;
    invJ[2][0] =  (J[1][0] * J[2][1] - J[2][0] * J[1][1]) * inv_det;
    invJ[2][1] = -(J[0][0] * J[2][1] - J[2][0] * J[0][1]) * inv_det;
    invJ[2][2] =  (J[0][0] * J[1][1] - J[1][0] * J[0][1]) * inv_det;

    // Shape function derivatives in parametric space
    // dN/dξ = [-1, 1, 0, 0]
    // dN/dη = [-1, 0, 1, 0]
    // dN/dζ = [-1, 0, 0, 1]

    // Compute physical gradients: dN/dx = invJ^T * dN/dξ
    // Node 1: dN/dξ = [-1, -1, -1]
    shape_grads[0][0] = -invJ[0][0] - invJ[1][0] - invJ[2][0];
    shape_grads[0][1] = -invJ[0][1] - invJ[1][1] - invJ[2][1];
    shape_grads[0][2] = -invJ[0][2] - invJ[1][2] - invJ[2][2];

    // Node 2: dN/dξ = [1, 0, 0]
    shape_grads[1][0] = invJ[0][0];
    shape_grads[1][1] = invJ[0][1];
    shape_grads[1][2] = invJ[0][2];

    // Node 3: dN/dξ = [0, 1, 0]
    shape_grads[2][0] = invJ[1][0];
    shape_grads[2][1] = invJ[1][1];
    shape_grads[2][2] = invJ[1][2];

    // Node 4: dN/dξ = [0, 0, 1]
    shape_grads[3][0] = invJ[2][0];
    shape_grads[3][1] = invJ[2][1];
    shape_grads[3][2] = invJ[2][2];
}

void TetElement::BuildBMatrix(
    const std::array<std::array<double, 3>, 4>& shape_grads,
    std::array<std::array<double, 12>, 6>& B) {

    // Initialize to zero
    for (auto& row : B) {
        row.fill(0.0);
    }

    // Fill B matrix
    for (int i = 0; i < 4; ++i) {
        const int col = i * 3;
        const double gx = shape_grads[i][0];
        const double gy = shape_grads[i][1];
        const double gz = shape_grads[i][2];

        // ε_xx = ∂u/∂x
        B[0][col + 0] = gx;

        // ε_yy = ∂v/∂y
        B[1][col + 1] = gy;

        // ε_zz = ∂w/∂z
        B[2][col + 2] = gz;

        // γ_xy = ∂u/∂y + ∂v/∂x
        B[3][col + 0] = gy;
        B[3][col + 1] = gx;

        // γ_xz = ∂u/∂z + ∂w/∂x
        B[4][col + 0] = gz;
        B[4][col + 2] = gx;

        // γ_yz = ∂v/∂z + ∂w/∂y
        B[5][col + 1] = gz;
        B[5][col + 2] = gy;
    }
}

} // namespace femml
