#include "element/HexElement.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace femml {

HexElement::HexElement(int id, const std::vector<int>& node_ids,
                       const std::shared_ptr<Material>& material)
    : Element(id, node_ids, material) {
    if (node_ids.size() != 8) {
        throw std::invalid_argument("Hex element must have 8 nodes");
    }
}

void HexElement::ComputeInternalForce(
    const std::vector<std::array<double, 3>>& node_coords,
    const std::vector<std::array<double, 3>>& node_displacements,
    std::vector<std::array<double, 3>>& node_forces) {

    if (node_coords.size() != 8 || node_displacements.size() != 8 || node_forces.size() != 8) {
        throw std::invalid_argument("HexElement: incorrect array sizes");
    }

    // Compute shape function gradients and Jacobian determinant
    std::array<std::array<double, 3>, 8> shape_grads;
    double det_jac;
    ComputeShapeGradients(node_coords, shape_grads, det_jac);

    // Volume for single Gauss point integration
    double volume = std::abs(det_jac) * 8.0;  // Weight = 8 for center point

    // Build B matrix (6x24)
    std::array<std::array<double, 24>, 6> B;
    BuildBMatrix(shape_grads, B);

    // Flatten displacements to vector [u1x, u1y, u1z, u2x, ...]
    std::array<double, 24> u_flat{};
    for (int i = 0; i < 8; ++i) {
        u_flat[i * 3 + 0] = node_displacements[i][0];
        u_flat[i * 3 + 1] = node_displacements[i][1];
        u_flat[i * 3 + 2] = node_displacements[i][2];
    }

    // Compute strain: ε = B * u
    std::array<double, 6> new_strain{};
    for (int i = 0; i < 6; ++i) {
        double val = 0.0;
        for (int j = 0; j < 24; ++j) {
            val += B[i][j] * u_flat[j];
        }
        new_strain[i] = val;
    }

    // Strain increment drives Abaqus-style constitutive update
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
    std::array<double, 24> f_int_flat{};
    for (int j = 0; j < 24; ++j) {
        double val = 0.0;
        for (int i = 0; i < 6; ++i) {
            val += B[i][j] * stress_[i];
        }
        f_int_flat[j] = val * volume;
    }

    // Unflatten to node forces
    for (int i = 0; i < 8; ++i) {
        node_forces[i][0] = f_int_flat[i * 3 + 0];
        node_forces[i][1] = f_int_flat[i * 3 + 1];
        node_forces[i][2] = f_int_flat[i * 3 + 2];
    }
}

void HexElement::ComputeMassMatrix(
    const std::vector<std::array<double, 3>>& node_coords,
    std::vector<double>& lumped_mass) {

    if (lumped_mass.size() != 8) {
        lumped_mass.resize(8);
    }

    double volume = ComputeVolume(node_coords);
    double total_mass = material_->GetDensity() * volume;
    double mass_per_node = total_mass / 8.0;

    for (int i = 0; i < 8; ++i) {
        lumped_mass[i] = mass_per_node;
    }
}

double HexElement::ComputeVolume(
    const std::vector<std::array<double, 3>>& node_coords) {

    std::array<std::array<double, 3>, 8> shape_grads;
    double det_jac;
    ComputeShapeGradients(node_coords, shape_grads, det_jac);

    return std::abs(det_jac) * 8.0;
}

double HexElement::ComputeCriticalTimeStep(
    const std::vector<std::array<double, 3>>& node_coords) {

    double char_length = ComputeCharacteristicLength(node_coords);
    double rho = material_->GetDensity();

    // Get wave speed from material
    // For linear elastic: c = sqrt(E / rho)
    // For now, assume we can extract E from material
    // TODO: Add method to Material class to get wave speed
    double E = 70e9;  // Default, should get from material
    double wave_speed = std::sqrt(E / rho);

    return char_length / wave_speed;
}

void HexElement::ComputeShapeGradients(
    const std::vector<std::array<double, 3>>& node_coords,
    std::array<std::array<double, 3>, 8>& shape_grads,
    double& det_jacobian) {

    // Evaluate at element center (ξ=η=ζ=0)
    const double xi = 0.0;
    const double eta = 0.0;
    const double zeta = 0.0;

    // Node positions in parametric space
    const int s[8][3] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}
    };

    // Compute shape function derivatives in parametric space
    std::array<std::array<double, 3>, 8> dN_dxi{};
    for (int i = 0; i < 8; ++i) {
        double sx = static_cast<double>(s[i][0]);
        double sy = static_cast<double>(s[i][1]);
        double sz = static_cast<double>(s[i][2]);

        dN_dxi[i][0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dN_dxi[i][1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
        dN_dxi[i][2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
    }

    // Compute Jacobian matrix J = dN_dxi^T * coords
    std::array<std::array<double, 3>, 3> J{};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            double val = 0.0;
            for (int i = 0; i < 8; ++i) {
                val += dN_dxi[i][a] * node_coords[i][b];
            }
            J[a][b] = val;
        }
    }

    // Compute determinant
    det_jacobian = J[0][0] * (J[1][1] * J[2][2] - J[2][1] * J[1][2]) -
                   J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2]) +
                   J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]);

    if (std::abs(det_jacobian) < 1e-12) {
        throw std::runtime_error("Hex element has zero or negative Jacobian");
    }

    // Compute inverse Jacobian
    double inv_det = 1.0 / det_jacobian;
    std::array<std::array<double, 3>, 3> invJ{};
    invJ[0][0] =  (J[1][1] * J[2][2] - J[2][1] * J[1][2]) * inv_det;
    invJ[0][1] = -(J[0][1] * J[2][2] - J[2][1] * J[0][2]) * inv_det;
    invJ[0][2] =  (J[0][1] * J[1][2] - J[1][1] * J[0][2]) * inv_det;
    invJ[1][0] = -(J[1][0] * J[2][2] - J[2][0] * J[1][2]) * inv_det;
    invJ[1][1] =  (J[0][0] * J[2][2] - J[2][0] * J[0][2]) * inv_det;
    invJ[1][2] = -(J[0][0] * J[1][2] - J[1][0] * J[0][2]) * inv_det;
    invJ[2][0] =  (J[1][0] * J[2][1] - J[2][0] * J[1][1]) * inv_det;
    invJ[2][1] = -(J[0][0] * J[2][1] - J[2][0] * J[0][1]) * inv_det;
    invJ[2][2] =  (J[0][0] * J[1][1] - J[1][0] * J[0][1]) * inv_det;

    // Compute dN_dx = invJ^T * dN_dxi
    for (int i = 0; i < 8; ++i) {
        for (int a = 0; a < 3; ++a) {
            shape_grads[i][a] = invJ[0][a] * dN_dxi[i][0] +
                                invJ[1][a] * dN_dxi[i][1] +
                                invJ[2][a] * dN_dxi[i][2];
        }
    }
}

void HexElement::BuildBMatrix(
    const std::array<std::array<double, 3>, 8>& shape_grads,
    std::array<std::array<double, 24>, 6>& B) {

    // Initialize to zero
    for (auto& row : B) {
        row.fill(0.0);
    }

    // Fill B matrix
    // Strain: [ε_xx, ε_yy, ε_zz, γ_xy, γ_xz, γ_yz]
    // B relates strain to nodal displacements
    for (int i = 0; i < 8; ++i) {
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

double HexElement::ComputeCharacteristicLength(
    const std::vector<std::array<double, 3>>& node_coords) {

    // Use minimum edge length
    double min_length = std::numeric_limits<double>::max();

    // Check all 12 edges
    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
    };

    for (int e = 0; e < 12; ++e) {
        int n1 = edges[e][0];
        int n2 = edges[e][1];

        double dx = node_coords[n2][0] - node_coords[n1][0];
        double dy = node_coords[n2][1] - node_coords[n1][1];
        double dz = node_coords[n2][2] - node_coords[n1][2];

        double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        min_length = std::min(min_length, length);
    }

    return min_length;
}

} // namespace femml
