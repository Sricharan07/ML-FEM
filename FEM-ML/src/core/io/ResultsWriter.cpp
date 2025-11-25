#include "io/ResultsWriter.hpp"
#include <iostream>
#include <iomanip>

namespace femml {

ResultsWriter::ResultsWriter(const std::string& filename, OutputFormat format)
    : filename_(filename), format_(format), mesh_written_(false), num_steps_written_(0) {
}

ResultsWriter::~ResultsWriter() {
    Finalize();
}

void ResultsWriter::WriteMesh(const std::shared_ptr<Mesh>& mesh) {
    mesh_ = mesh;
    mesh_written_ = true;
}

void ResultsWriter::WriteStep(
    int step,
    double time,
    const std::vector<std::array<double, 3>>& displacements,
    const std::vector<std::array<double, 3>>& velocities,
    const std::vector<std::shared_ptr<Element>>& elements) {

    switch (format_) {
        case OutputFormat::CSV:
            WriteCSV(step, time, displacements, velocities, elements);
            break;

        case OutputFormat::VTU:
            WriteVTU(step, time, displacements, velocities, elements);
            break;

        default:
            std::cerr << "Output format not implemented" << std::endl;
            break;
    }

    num_steps_written_++;
}

void ResultsWriter::WriteCSV(
    int step,
    double time,
    const std::vector<std::array<double, 3>>& displacements,
    const std::vector<std::array<double, 3>>& velocities,
    const std::vector<std::shared_ptr<Element>>& elements) {

    if (!file_.is_open()) {
        file_.open(filename_);
        if (!file_) {
            throw std::runtime_error("Failed to open output file: " + filename_);
        }

        // Write header
        file_ << "Step,Time,NodeID,X,Y,Z,Ux,Uy,Uz,Vx,Vy,Vz\n";
    }

    // Write node results
    int node_idx = 0;
    for (const auto& [node_id, node] : mesh_->GetAllNodes()) {
        file_ << step << "," << std::scientific << std::setprecision(6) << time << ","
              << node_id << ","
              << node.coords[0] << "," << node.coords[1] << "," << node.coords[2] << ","
              << displacements[node_idx][0] << "," << displacements[node_idx][1] << "," << displacements[node_idx][2] << ","
              << velocities[node_idx][0] << "," << velocities[node_idx][1] << "," << velocities[node_idx][2] << "\n";
        node_idx++;
    }
}

void ResultsWriter::WriteVTU(
    int step,
    double time,
    const std::vector<std::array<double, 3>>& displacements,
    const std::vector<std::array<double, 3>>& velocities,
    const std::vector<std::shared_ptr<Element>>& elements) {

    // Generate filename with step number
    std::string vtu_filename = filename_;
    size_t dot_pos = vtu_filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        vtu_filename = vtu_filename.substr(0, dot_pos) + "_" +
                       std::to_string(step) + ".vtu";
    }
    else {
        vtu_filename += "_" + std::to_string(step) + ".vtu";
    }

    std::ofstream vtu_file(vtu_filename);
    if (!vtu_file) {
        throw std::runtime_error("Failed to open VTU file: " + vtu_filename);
    }

    // Write VTK XML header
    vtu_file << "<?xml version=\"1.0\"?>\n";
    vtu_file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtu_file << "  <UnstructuredGrid>\n";
    vtu_file << "    <Piece NumberOfPoints=\"" << mesh_->GetNumNodes()
             << "\" NumberOfCells=\"" << mesh_->GetNumElements() << "\">\n";

    // TODO: Complete VTU implementation
    // This is a placeholder for now

    vtu_file << "    </Piece>\n";
    vtu_file << "  </UnstructuredGrid>\n";
    vtu_file << "</VTKFile>\n";

    vtu_file.close();
}

void ResultsWriter::WriteField(
    const std::string& name,
    FieldType type,
    const std::vector<double>& scalar_data) {
    // TODO: Implement field writing
}

void ResultsWriter::WriteField(
    const std::string& name,
    FieldType type,
    const std::vector<std::array<double, 3>>& vector_data) {
    // TODO: Implement field writing
}

void ResultsWriter::Finalize() {
    if (file_.is_open()) {
        file_.close();
        std::cout << "Results file closed: " << filename_ << std::endl;
        std::cout << "Total steps written: " << num_steps_written_ << std::endl;
    }
}

// History writer implementation
HistoryWriter::HistoryWriter(const std::string& filename)
    : filename_(filename) {
    file_.open(filename);
    if (!file_) {
        throw std::runtime_error("Failed to open history file: " + filename);
    }
}

HistoryWriter::~HistoryWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

void HistoryWriter::SetNodes(const std::vector<int>& node_ids) {
    node_ids_ = node_ids;
}

void HistoryWriter::WriteHeader() {
    file_ << "Time";
    for (int node_id : node_ids_) {
        file_ << ",Ux" << node_id << ",Uy" << node_id << ",Uz" << node_id;
    }
    file_ << "\n";
}

void HistoryWriter::WriteStep(
    double time,
    const std::map<int, std::array<double, 3>>& node_displacements) {

    file_ << std::scientific << std::setprecision(10) << time;

    for (int node_id : node_ids_) {
        auto it = node_displacements.find(node_id);
        if (it != node_displacements.end()) {
            file_ << "," << it->second[0]
                  << "," << it->second[1]
                  << "," << it->second[2];
        }
        else {
            file_ << ",0.0,0.0,0.0";
        }
    }

    file_ << "\n";
}

} // namespace femml
