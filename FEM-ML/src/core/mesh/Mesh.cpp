#include "mesh/Mesh.hpp"
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace femml {

const Node& Mesh::GetNode(int id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        throw std::runtime_error("Node " + std::to_string(id) + " not found");
    }
    return it->second;
}

Node& Mesh::GetNode(int id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        throw std::runtime_error("Node " + std::to_string(id) + " not found");
    }
    return it->second;
}

const ElementConnectivity& Mesh::GetElement(int id) const {
    auto it = elements_.find(id);
    if (it == elements_.end()) {
        throw std::runtime_error("Element " + std::to_string(id) + " not found");
    }
    return it->second;
}

ElementConnectivity& Mesh::GetElement(int id) {
    auto it = elements_.find(id);
    if (it == elements_.end()) {
        throw std::runtime_error("Element " + std::to_string(id) + " not found");
    }
    return it->second;
}

void Mesh::AddNode(const Node& node) {
    nodes_[node.id] = node;
}

void Mesh::AddElement(const ElementConnectivity& elem) {
    elements_[elem.id] = elem;
}

void Mesh::AddNodeSet(const std::string& name, const std::vector<int>& nodes) {
    NodeSet ns;
    ns.name = name;
    ns.nodes = nodes;
    node_sets_[name] = ns;
}

void Mesh::AddElementSet(const std::string& name, const std::vector<int>& elements) {
    ElementSet es;
    es.name = name;
    es.elements = elements;
    element_sets_[name] = es;
}

void Mesh::AddSurface(const std::string& name,
                      const std::vector<std::pair<int, int>>& faces) {
    Surface surf;
    surf.name = name;
    surf.faces = faces;
    surfaces_[name] = surf;
}

const NodeSet* Mesh::GetNodeSet(const std::string& name) const {
    auto it = node_sets_.find(name);
    return (it != node_sets_.end()) ? &it->second : nullptr;
}

const ElementSet* Mesh::GetElementSet(const std::string& name) const {
    auto it = element_sets_.find(name);
    return (it != element_sets_.end()) ? &it->second : nullptr;
}

const Surface* Mesh::GetSurface(const std::string& name) const {
    auto it = surfaces_.find(name);
    return (it != surfaces_.end()) ? &it->second : nullptr;
}

std::array<double, 3> Mesh::GetMinCoords() const {
    if (nodes_.empty()) {
        return {0.0, 0.0, 0.0};
    }

    std::array<double, 3> min_coords = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    };

    for (const auto& [id, node] : nodes_) {
        for (int i = 0; i < 3; ++i) {
            min_coords[i] = std::min(min_coords[i], node.coords[i]);
        }
    }

    return min_coords;
}

std::array<double, 3> Mesh::GetMaxCoords() const {
    if (nodes_.empty()) {
        return {0.0, 0.0, 0.0};
    }

    std::array<double, 3> max_coords = {
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
    };

    for (const auto& [id, node] : nodes_) {
        for (int i = 0; i < 3; ++i) {
            max_coords[i] = std::max(max_coords[i], node.coords[i]);
        }
    }

    return max_coords;
}

void Mesh::Clear() {
    nodes_.clear();
    elements_.clear();
    node_sets_.clear();
    element_sets_.clear();
    surfaces_.clear();
}

void Mesh::PrintSummary() const {
    std::cout << "=== Mesh Summary ===" << std::endl;
    std::cout << "Nodes: " << nodes_.size() << std::endl;
    std::cout << "Elements: " << elements_.size() << std::endl;
    std::cout << "Node sets: " << node_sets_.size() << std::endl;
    std::cout << "Element sets: " << element_sets_.size() << std::endl;
    std::cout << "Surfaces: " << surfaces_.size() << std::endl;

    auto min_coords = GetMinCoords();
    auto max_coords = GetMaxCoords();
    std::cout << "Bounding box: " << std::endl;
    std::cout << "  Min: (" << min_coords[0] << ", " << min_coords[1] << ", " << min_coords[2] << ")" << std::endl;
    std::cout << "  Max: (" << max_coords[0] << ", " << max_coords[1] << ", " << max_coords[2] << ")" << std::endl;
    std::cout << "===================" << std::endl;
}

} // namespace femml
