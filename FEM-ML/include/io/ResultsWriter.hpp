#pragma once

#include "mesh/Mesh.hpp"
#include "element/Element.hpp"
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <fstream>

namespace femml {

// Output formats
enum class OutputFormat {
    CSV,      // Simple CSV format
    VTU,      // VTK unstructured grid (XML)
    HDF5,     // HDF5 format (requires dependency)
    ABAQUS    // Abaqus ODB-like text format
};

// Field variable types
enum class FieldType {
    DISPLACEMENT,
    VELOCITY,
    ACCELERATION,
    STRESS,
    STRAIN,
    ENERGY
};

// Results writer for post-processing
class ResultsWriter {
public:
    ResultsWriter(const std::string& filename, OutputFormat format = OutputFormat::CSV);
    ~ResultsWriter();

    // Write mesh (once at beginning)
    void WriteMesh(const std::shared_ptr<Mesh>& mesh);

    // Write time step results
    void WriteStep(
        int step,
        double time,
        const std::vector<std::array<double, 3>>& displacements,
        const std::vector<std::array<double, 3>>& velocities,
        const std::vector<std::shared_ptr<Element>>& elements);

    // Write specific field
    void WriteField(
        const std::string& name,
        FieldType type,
        const std::vector<double>& scalar_data);

    void WriteField(
        const std::string& name,
        FieldType type,
        const std::vector<std::array<double, 3>>& vector_data);

    // Finalize and close file
    void Finalize();

private:
    void WriteCSV(int step, double time,
                  const std::vector<std::array<double, 3>>& displacements,
                  const std::vector<std::array<double, 3>>& velocities,
                  const std::vector<std::shared_ptr<Element>>& elements);

    void WriteVTU(int step, double time,
                  const std::vector<std::array<double, 3>>& displacements,
                  const std::vector<std::array<double, 3>>& velocities,
                  const std::vector<std::shared_ptr<Element>>& elements);

    std::string filename_;
    OutputFormat format_;
    std::ofstream file_;
    std::shared_ptr<Mesh> mesh_;
    bool mesh_written_;
    int num_steps_written_;
};

// History writer for time-series data at specific nodes
class HistoryWriter {
public:
    HistoryWriter(const std::string& filename);
    ~HistoryWriter();

    void SetNodes(const std::vector<int>& node_ids);
    void WriteHeader();
    void WriteStep(double time,
                   const std::map<int, std::array<double, 3>>& node_displacements);

private:
    std::string filename_;
    std::ofstream file_;
    std::vector<int> node_ids_;
};

} // namespace femml
