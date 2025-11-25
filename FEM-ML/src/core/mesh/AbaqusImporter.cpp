#include "mesh/AbaqusImporter.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace femml {

std::shared_ptr<Mesh> AbaqusImporter::Import(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    auto mesh = std::make_shared<Mesh>();
    std::string line;
    pending_line_.clear();
    has_pending_line_ = false;

    std::cout << "Importing Abaqus file: " << filename << std::endl;

    while (GetNextLine(file, line)) {
        std::string upper_line = ToUpper(line);

        if (upper_line.find("*NODE") == 0 || upper_line.find("*NODAL") == 0) {
            ParseNodes(file, *mesh);
        }
        else if (upper_line.find("*ELEMENT") == 0) {
            // Extract element type
            std::string elem_type = "C3D8";  // Default
            size_t type_pos = upper_line.find("TYPE=");
            if (type_pos != std::string::npos) {
                size_t start = type_pos + 5;
                size_t end = upper_line.find(',', start);
                if (end == std::string::npos) end = upper_line.length();
                elem_type = Trim(upper_line.substr(start, end - start));
            }
            ParseElements(file, *mesh, elem_type);
        }
        else if (upper_line.find("*NSET") == 0) {
            // Extract set name
            size_t nset_pos = upper_line.find("NSET=");
            if (nset_pos != std::string::npos) {
                size_t start = nset_pos + 5;
                size_t end = upper_line.find(',', start);
                if (end == std::string::npos) end = upper_line.length();
                std::string set_name = Trim(upper_line.substr(start, end - start));
                ParseNodeSet(file, *mesh, set_name);
            }
        }
        else if (upper_line.find("*ELSET") == 0) {
            size_t elset_pos = upper_line.find("ELSET=");
            if (elset_pos != std::string::npos) {
                size_t start = elset_pos + 6;
                size_t end = upper_line.find(',', start);
                if (end == std::string::npos) end = upper_line.length();
                std::string set_name = Trim(upper_line.substr(start, end - start));
                ParseElementSet(file, *mesh, set_name);
            }
        }
        else if (upper_line.find("*SURFACE") == 0) {
            size_t surf_pos = upper_line.find("NAME=");
            if (surf_pos != std::string::npos) {
                size_t start = surf_pos + 5;
                size_t end = upper_line.find(',', start);
                if (end == std::string::npos) end = upper_line.length();
                std::string surf_name = Trim(upper_line.substr(start, end - start));
                ParseSurface(file, *mesh, surf_name);
            }
        }
    }

    std::cout << "Import completed." << std::endl;
    mesh->PrintSummary();

    return mesh;
}

void AbaqusImporter::ParseNodes(std::istream& file, Mesh& mesh) {
    std::string line;
    int node_count = 0;

    while (GetNextLine(file, line)) {
        if (line.empty()) {
            continue;
        }
        char c = line[0];
        if (c == '*') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }
        if (!std::isdigit(c) && c != '-' && c != '+') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }

        auto parts = Split(line, ',');
        if (parts.size() >= 4) {
            int id = std::stoi(Trim(parts[0]));
            double x = std::stod(Trim(parts[1]));
            double y = std::stod(Trim(parts[2]));
            double z = std::stod(Trim(parts[3]));

            mesh.AddNode(Node(id, x, y, z));
            node_count++;
        }
    }

    std::cout << "  Parsed " << node_count << " nodes" << std::endl;
}

void AbaqusImporter::ParseElements(std::istream& file, Mesh& mesh, const std::string& type) {
    std::string line;
    int elem_count = 0;

    while (GetNextLine(file, line)) {
        if (line.empty()) {
            continue;
        }
        char c = line[0];
        if (c == '*') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }
        if (!std::isdigit(c) && c != '-' && c != '+') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }

        auto parts = Split(line, ',');
        if (parts.size() >= 2) {
            int id = std::stoi(Trim(parts[0]));
            std::vector<int> nodes;
            for (size_t i = 1; i < parts.size(); ++i) {
                std::string node_str = Trim(parts[i]);
                if (!node_str.empty()) {
                    nodes.push_back(std::stoi(node_str));
                }
            }

            // If line ends with comma, continue reading
            if (line.back() == ',') {
                std::string continuation;
                if (std::getline(file, continuation)) {
                    auto cont_parts = Split(continuation, ',');
                    for (const auto& part : cont_parts) {
                        std::string node_str = Trim(part);
                        if (!node_str.empty()) {
                            nodes.push_back(std::stoi(node_str));
                        }
                    }
                }
            }

            mesh.AddElement(ElementConnectivity(id, type, nodes));
            elem_count++;
        }
    }

    std::cout << "  Parsed " << elem_count << " elements (type: " << type << ")" << std::endl;
}

void AbaqusImporter::ParseNodeSet(std::istream& file, Mesh& mesh, const std::string& name) {
    std::string line;
    std::vector<int> nodes;

    while (GetNextLine(file, line)) {
        if (line.empty()) {
            continue;
        }
        char c = line[0];
        if (c == '*') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }
        if (!std::isdigit(c) && c != '-' && c != '+') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }

        auto parts = Split(line, ',');
        for (const auto& part : parts) {
            std::string node_str = Trim(part);
            if (!node_str.empty()) {
                nodes.push_back(std::stoi(node_str));
            }
        }
    }

    mesh.AddNodeSet(name, nodes);
    std::cout << "  Parsed node set '" << name << "' with " << nodes.size() << " nodes" << std::endl;
}

void AbaqusImporter::ParseElementSet(std::istream& file, Mesh& mesh, const std::string& name) {
    std::string line;
    std::vector<int> elements;

    while (GetNextLine(file, line)) {
        if (line.empty()) {
            continue;
        }
        char c = line[0];
        if (c == '*') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }
        if (!std::isdigit(c) && c != '-' && c != '+') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }

        auto parts = Split(line, ',');
        for (const auto& part : parts) {
            std::string elem_str = Trim(part);
            if (!elem_str.empty()) {
                elements.push_back(std::stoi(elem_str));
            }
        }
    }

    mesh.AddElementSet(name, elements);
    std::cout << "  Parsed element set '" << name << "' with " << elements.size() << " elements" << std::endl;
}

void AbaqusImporter::ParseSurface(std::istream& file, Mesh& mesh, const std::string& name) {
    std::string line;
    std::vector<std::pair<int, int>> faces;

    while (GetNextLine(file, line)) {
        if (line.empty()) {
            continue;
        }
        char c = line[0];
        if (c == '*') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }
        if (!std::isdigit(c) && c != '-' && c != '+') {
            pending_line_ = line;
            has_pending_line_ = true;
            break;
        }

        auto parts = Split(line, ',');
        if (parts.size() >= 2) {
            int elem_id = std::stoi(Trim(parts[0]));
            std::string face_str = Trim(parts[1]);
            int face_id = (face_str[0] == 'S') ? std::stoi(face_str.substr(1)) : std::stoi(face_str);
            faces.push_back({elem_id, face_id});
        }
    }

    mesh.AddSurface(name, faces);
    std::cout << "  Parsed surface '" << name << "' with " << faces.size() << " faces" << std::endl;
}

bool AbaqusImporter::GetNextLine(std::istream& file, std::string& line) {
    if (has_pending_line_) {
        line = pending_line_;
        has_pending_line_ = false;
        return true;
    }

    std::string raw;
    while (std::getline(file, raw)) {
        std::string trimmed = Trim(raw);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.rfind("**", 0) == 0) {
            continue;
        }
        line = trimmed;
        return true;
    }

    line.clear();
    return false;
}

std::string AbaqusImporter::Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::string AbaqusImporter::ToUpper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::vector<std::string> AbaqusImporter::Split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace femml
