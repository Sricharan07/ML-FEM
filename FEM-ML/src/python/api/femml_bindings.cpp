#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "mesh/Mesh.hpp"
#include "mesh/AbaqusImporter.hpp"
#include "material/LinearElastic.hpp"
#include "material/NeuralMaterial.hpp"
#include "element/Element.hpp"
#include "element/HexElement.hpp"
#include "element/TetElement.hpp"
#ifdef FEMML_USE_GPU
#include "solver/GPUExplicitSolver.hpp"
#endif
#include "solver/ExplicitSolver.hpp"
#include "io/ResultsWriter.hpp"

namespace py = pybind11;
using namespace femml;

PYBIND11_MODULE(femml, m) {
    m.doc() = "FEM-ML: Explicit FEM Solver with Neural Network Integration";

    // Mesh classes
    py::class_<Node>(m, "Node")
        .def(py::init<>())
        .def(py::init<int, double, double, double>())
        .def_readwrite("id", &Node::id)
        .def_readwrite("coords", &Node::coords);

    py::class_<ElementConnectivity>(m, "ElementConnectivity")
        .def(py::init<>())
        .def(py::init<int, const std::string&, const std::vector<int>&>())
        .def_readwrite("id", &ElementConnectivity::id)
        .def_readwrite("type", &ElementConnectivity::type)
        .def_readwrite("nodes", &ElementConnectivity::nodes);

    py::class_<Mesh, std::shared_ptr<Mesh>>(m, "Mesh")
        .def(py::init<>())
        .def("add_node", &Mesh::AddNode)
        .def("add_element", &Mesh::AddElement)
        .def("add_node_set", &Mesh::AddNodeSet)
        .def("add_element_set", &Mesh::AddElementSet)
        .def("get_num_nodes", &Mesh::GetNumNodes)
        .def("get_num_elements", &Mesh::GetNumElements)
        .def("get_node", (const Node& (Mesh::*)(int) const) &Mesh::GetNode)
        .def("get_element", (const ElementConnectivity& (Mesh::*)(int) const) &Mesh::GetElement)
        .def("print_summary", &Mesh::PrintSummary)
        .def("clear", &Mesh::Clear);

    py::class_<AbaqusImporter>(m, "AbaqusImporter")
        .def(py::init<>())
        .def("import_mesh", &AbaqusImporter::Import,
             "Import Abaqus .inp file",
             py::arg("filename"));

    // Material classes
    py::class_<Material, std::shared_ptr<Material>>(m, "Material")
        .def("compute_stress_increment", &Material::ComputeStressIncrement)
        .def("get_name", &Material::GetName)
        .def("get_density", &Material::GetDensity)
        .def("set_density", &Material::SetDensity)
        .def("get_type", &Material::GetType);

    py::class_<LinearElastic, Material, std::shared_ptr<LinearElastic>>(m, "LinearElastic")
        .def(py::init<const std::string&, double, double, double>(),
             py::arg("name"), py::arg("E"), py::arg("nu"), py::arg("rho"))
        .def("get_youngs_modulus", &LinearElastic::GetYoungsModulus)
        .def("get_poissons_ratio", &LinearElastic::GetPoissonsRatio);

    m.def("create_aluminum", &CreateAluminum, "Create aluminum material");
    m.def("create_steel", &CreateSteel, "Create steel material");
    m.def("create_titanium", &CreateTitanium, "Create titanium material");

    py::class_<NeuralMaterial, Material, std::shared_ptr<NeuralMaterial>>(m, "NeuralMaterial")
        .def(py::init<const std::string&, double>(),
             py::arg("name"), py::arg("rho"))
        .def("set_inference_callback", &NeuralMaterial::SetInferenceCallback)
        .def("set_fallback_material", &NeuralMaterial::SetFallbackMaterial)
        .def("load_onnx_model", &NeuralMaterial::LoadONNXModel);

    // Solver classes
    py::enum_<BCType>(m, "BCType")
        .value("FIXED", BCType::FIXED)
        .value("DISPLACEMENT", BCType::DISPLACEMENT)
        .value("VELOCITY", BCType::VELOCITY)
        .value("ACCELERATION", BCType::ACCELERATION);

    py::enum_<LoadType>(m, "LoadType")
        .value("FORCE", LoadType::FORCE)
        .value("PRESSURE", LoadType::PRESSURE)
        .value("BODY_FORCE", LoadType::BODY_FORCE)
        .value("GRAVITY", LoadType::GRAVITY);

    py::class_<BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("nodes", &BoundaryCondition::nodes)
        .def_readwrite("type", &BoundaryCondition::type)
        .def_readwrite("component", &BoundaryCondition::component)
        .def_readwrite("value", &BoundaryCondition::value);

    py::class_<Load>(m, "Load")
        .def(py::init<>())
        .def_readwrite("type", &Load::type)
        .def_readwrite("nodes", &Load::nodes)
        .def_readwrite("surface", &Load::surface)
        .def_readwrite("component", &Load::component)
        .def_readwrite("value", &Load::value);

    py::class_<SolverParams>(m, "SolverParams")
        .def(py::init<>())
        .def_readwrite("time_step", &SolverParams::time_step)
        .def_readwrite("num_steps", &SolverParams::num_steps)
        .def_readwrite("damping", &SolverParams::damping)
        .def_readwrite("output_interval", &SolverParams::output_interval)
        .def_readwrite("auto_time_step", &SolverParams::auto_time_step)
        .def_readwrite("time_step_scale", &SolverParams::time_step_scale);

    py::class_<ExplicitSolver::NodeResult>(m, "NodeResult")
        .def_readonly("id", &ExplicitSolver::NodeResult::id)
        .def_readonly("coords", &ExplicitSolver::NodeResult::coords)
        .def_readonly("displacement", &ExplicitSolver::NodeResult::displacement);

    py::class_<ExplicitSolver::ElementResult>(m, "ElementResult")
        .def_readonly("id", &ExplicitSolver::ElementResult::id)
        .def_readonly("type", &ExplicitSolver::ElementResult::type)
        .def_readonly("volume", &ExplicitSolver::ElementResult::volume)
        .def_readonly("stress", &ExplicitSolver::ElementResult::stress)
        .def_readonly("strain", &ExplicitSolver::ElementResult::strain);

    py::class_<ExplicitSolver>(m, "ExplicitSolver")
        .def(py::init<const std::shared_ptr<Mesh>&>())
        .def("set_material", (void (ExplicitSolver::*)(const std::shared_ptr<Material>&)) &ExplicitSolver::SetMaterial,
             "Set material for all elements")
        .def("set_material_for_set", (void (ExplicitSolver::*)(const std::string&, const std::shared_ptr<Material>&)) &ExplicitSolver::SetMaterial,
             "Set material for specific element set")
        .def("add_boundary_condition", &ExplicitSolver::AddBoundaryCondition)
        .def("add_load", &ExplicitSolver::AddLoad)
        .def("set_parameters", &ExplicitSolver::SetParameters)
        .def("initialize", &ExplicitSolver::Initialize)
        .def("solve", &ExplicitSolver::Solve)
        .def("step", &ExplicitSolver::Step)
        .def("is_finished", &ExplicitSolver::IsFinished)
        .def("get_current_step", &ExplicitSolver::GetCurrentStep)
        .def("get_current_time", &ExplicitSolver::GetCurrentTime)
        .def("get_total_steps", &ExplicitSolver::GetTotalSteps)
        .def("get_time_step_size", &ExplicitSolver::GetTimeStepSize)
        .def("get_displacements", &ExplicitSolver::GetDisplacements,
             py::return_value_policy::reference_internal)
        .def("get_velocities", &ExplicitSolver::GetVelocities,
             py::return_value_policy::reference_internal)
        .def("get_node_displacement", &ExplicitSolver::GetNodeDisplacement)
        .def("get_node_velocity", &ExplicitSolver::GetNodeVelocity)
        .def("get_node_results", &ExplicitSolver::GetNodeResults)
        .def("get_element_results", &ExplicitSolver::GetElementResults)
        .def("get_kinetic_energy", &ExplicitSolver::GetKineticEnergy)
        .def("get_strain_energy", &ExplicitSolver::GetStrainEnergy)
        .def("get_total_energy", &ExplicitSolver::GetTotalEnergy)
        .def("set_output_callback", &ExplicitSolver::SetOutputCallback)
        .def("write_results", &ExplicitSolver::WriteResults);

    // I/O classes
    py::enum_<OutputFormat>(m, "OutputFormat")
        .value("CSV", OutputFormat::CSV)
        .value("VTU", OutputFormat::VTU)
        .value("HDF5", OutputFormat::HDF5)
        .value("ABAQUS", OutputFormat::ABAQUS);

#ifdef FEMML_USE_GPU
    // GPU Solver bindings
    py::class_<GPUExplicitSolver, ExplicitSolver>(m, "GPUExplicitSolver")
        .def(py::init<const std::shared_ptr<Mesh>&>())
        .def("set_device_id", &GPUExplicitSolver::SetDeviceID, "Set CUDA device ID")
        .def("get_device_id", &GPUExplicitSolver::GetDeviceID, "Get current CUDA device ID")
        .def("get_gpu_compute_time", &GPUExplicitSolver::GetGPUComputeTime, "Get GPU computation time")
        .def("get_cpu_gpu_transfer_time", &GPUExplicitSolver::GetCPUGPUTransferTime, "Get data transfer time")
        .def("get_gpu_memory_usage", &GPUExplicitSolver::GetGPUMemoryUsage, "Get GPU memory usage in bytes");
#endif
    py::class_<HistoryWriter>(m, "HistoryWriter")
        .def(py::init<const std::string&>())
        .def("set_nodes", &HistoryWriter::SetNodes)
        .def("write_header", &HistoryWriter::WriteHeader)
        .def("write_step", &HistoryWriter::WriteStep);
}
