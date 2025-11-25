#pragma once

#include "mesh/Mesh.hpp"
#include <string>
#include <memory>

namespace femml {

class AbaqusImporter {
public:
    AbaqusImporter() = default;
    ~AbaqusImporter() = default;

    // Import Abaqus .inp file
    std::shared_ptr<Mesh> Import(const std::string& filename);

private:
    void ParseNodes(std::istream& file, Mesh& mesh);
    void ParseElements(std::istream& file, Mesh& mesh, const std::string& type);
    void ParseNodeSet(std::istream& file, Mesh& mesh, const std::string& name);
    void ParseElementSet(std::istream& file, Mesh& mesh, const std::string& name);
    void ParseSurface(std::istream& file, Mesh& mesh, const std::string& name);

    bool GetNextLine(std::istream& file, std::string& line);
    std::string Trim(const std::string& str);
    std::string ToUpper(const std::string& str);
    std::vector<std::string> Split(const std::string& str, char delimiter);

    std::string pending_line_;
    bool has_pending_line_ = false;
};

} // namespace femml
