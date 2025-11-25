#pragma once

#include <array>
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace femml {

// Node structure
struct Node {
    int id;
    std::array<double, 3> coords;

    Node() : id(0), coords{0.0, 0.0, 0.0} {}
    Node(int id_, double x, double y, double z)
        : id(id_), coords{x, y, z} {}
};

// Element connectivity
struct ElementConnectivity {
    int id;
    std::string type;  // C3D8, C3D8R, C3D4, etc.
    std::vector<int> nodes;  // Node IDs

    ElementConnectivity() : id(0), type("C3D8") {}
    ElementConnectivity(int id_, const std::string& type_, const std::vector<int>& nodes_)
        : id(id_), type(type_), nodes(nodes_) {}
};

// Node set for boundary conditions
struct NodeSet {
    std::string name;
    std::vector<int> nodes;
};

// Element set for sections/materials
struct ElementSet {
    std::string name;
    std::vector<int> elements;
};

// Surface for contact/loads
struct Surface {
    std::string name;
    std::vector<std::pair<int, int>> faces;  // (element_id, face_id)
};

// Main mesh class
class Mesh {
public:
    Mesh() = default;
    ~Mesh() = default;

    // Accessors
    int GetNumNodes() const { return static_cast<int>(nodes_.size()); }
    int GetNumElements() const { return static_cast<int>(elements_.size()); }

    const Node& GetNode(int id) const;
    Node& GetNode(int id);

    const ElementConnectivity& GetElement(int id) const;
    ElementConnectivity& GetElement(int id);

    // Add entities
    void AddNode(const Node& node);
    void AddElement(const ElementConnectivity& elem);
    void AddNodeSet(const std::string& name, const std::vector<int>& nodes);
    void AddElementSet(const std::string& name, const std::vector<int>& elements);
    void AddSurface(const std::string& name,
                    const std::vector<std::pair<int, int>>& faces);

    // Query
    const NodeSet* GetNodeSet(const std::string& name) const;
    const ElementSet* GetElementSet(const std::string& name) const;
    const Surface* GetSurface(const std::string& name) const;

    // Get all nodes/elements
    const std::map<int, Node>& GetAllNodes() const { return nodes_; }
    const std::map<int, ElementConnectivity>& GetAllElements() const { return elements_; }

    // Mesh info
    std::array<double, 3> GetMinCoords() const;
    std::array<double, 3> GetMaxCoords() const;

    // Clear
    void Clear();

    // I/O
    void PrintSummary() const;

private:
    std::map<int, Node> nodes_;
    std::map<int, ElementConnectivity> elements_;
    std::map<std::string, NodeSet> node_sets_;
    std::map<std::string, ElementSet> element_sets_;
    std::map<std::string, Surface> surfaces_;
};

} // namespace femml
