#include "mesh/Mesh.hpp"
#include "mesh/AbaqusImporter.hpp"
#include "material/LinearElastic.hpp"
#include "material/NeuralMaterial.hpp"
#include "solver/ExplicitSolver.hpp"
#include "io/ResultsWriter.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <sstream>

using namespace femml;

// Simple configuration file parser
struct Config {
    std::string mesh_file;
    std::string material_name = "aluminum";
    double E = 70e9;
    double nu = 0.33;
    double rho = 2700.0;
    double dt = 1e-7;
    int steps = 1000;
    double damping = 0.0;
    int output_interval = 10;
    bool auto_timestep = false;
    std::string output_file = "results.csv";
    std::vector<int> fixed_nodes;
    std::vector<std::tuple<int, int, double>> loads;  // node, component, value
};

Config ParseConfig(const std::string& filename) {
    Config config;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Warning: Config file not found: " << filename << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "mesh") {
            iss >> config.mesh_file;
        }
        else if (key == "material") {
            iss >> config.material_name;
        }
        else if (key == "E") {
            iss >> config.E;
        }
        else if (key == "nu") {
            iss >> config.nu;
        }
        else if (key == "rho") {
            iss >> config.rho;
        }
        else if (key == "dt") {
            iss >> config.dt;
        }
        else if (key == "steps") {
            iss >> config.steps;
        }
        else if (key == "damping") {
            iss >> config.damping;
        }
        else if (key == "output_interval") {
            iss >> config.output_interval;
        }
        else if (key == "auto_timestep") {
            std::string val;
            iss >> val;
            config.auto_timestep = (val == "true" || val == "1");
        }
        else if (key == "output") {
            iss >> config.output_file;
        }
        else if (key == "fixed") {
            int node;
            while (iss >> node) {
                config.fixed_nodes.push_back(node);
            }
        }
        else if (key == "force") {
            int node, comp;
            double value;
            iss >> node >> comp >> value;
            config.loads.push_back({node, comp, value});
        }
    }

    return config;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "    FEM-ML: Explicit FEM Solver v1.0    " << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Parse command line arguments
    std::string config_file = "config.inp";
    if (argc > 1) {
        config_file = argv[1];
    }

    // Load configuration
    Config config = ParseConfig(config_file);

    // Check if mesh file is specified
    if (config.mesh_file.empty()) {
        std::cerr << "Error: No mesh file specified in config" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    try {
        // Import mesh
        std::cout << "Importing mesh from: " << config.mesh_file << std::endl;
        AbaqusImporter importer;
        auto mesh = importer.Import(config.mesh_file);

        // Create material
        std::shared_ptr<Material> material;
        if (config.material_name == "aluminum") {
            material = CreateAluminum();
        }
        else if (config.material_name == "steel") {
            material = CreateSteel();
        }
        else if (config.material_name == "titanium") {
            material = CreateTitanium();
        }
        else {
            material = std::make_shared<LinearElastic>(
                config.material_name, config.E, config.nu, config.rho);
        }

        std::cout << "\nMaterial: " << material->GetName() << std::endl;
        std::cout << "  Type: " << material->GetType() << std::endl;
        std::cout << "  Density: " << material->GetDensity() << " kg/m³" << std::endl;

        // Create solver
        ExplicitSolver solver(mesh);

        // Assign material to all elements
        solver.SetMaterial(material);

        // Add boundary conditions
        if (!config.fixed_nodes.empty()) {
            BoundaryCondition bc;
            bc.type = BCType::FIXED;
            bc.nodes = config.fixed_nodes;
            bc.component = -1;  // All components
            bc.value = 0.0;
            solver.AddBoundaryCondition(bc);
            std::cout << "\nFixed nodes: " << config.fixed_nodes.size() << std::endl;
        }

        // Add loads
        for (const auto& [node, comp, value] : config.loads) {
            Load load;
            load.type = LoadType::FORCE;
            load.nodes = {node};
            load.component = comp;
            load.value = value;
            solver.AddLoad(load);
            std::cout << "Load: Node " << node << ", component " << comp
                      << ", value " << value << " N" << std::endl;
        }

        // Set solver parameters
        SolverParams params;
        params.time_step = config.dt;
        params.num_steps = config.steps;
        params.damping = config.damping;
        params.output_interval = config.output_interval;
        params.auto_time_step = config.auto_timestep;
        solver.SetParameters(params);

        // Initialize solver
        solver.Initialize();

        // Set output callback
        HistoryWriter history("history.csv");
        history.SetNodes({1, 2, 3, 4, 5, 6, 7, 8});  // Track first 8 nodes
        history.WriteHeader();

        solver.SetOutputCallback([&](int step, double time) {
            // Write history
            std::map<int, std::array<double, 3>> disps;
            for (int i = 1; i <= 8; ++i) {
                try {
                    disps[i] = solver.GetNodeDisplacement(i);
                }
                catch (...) {
                    // Node doesn't exist
                }
            }
            history.WriteStep(time, disps);

            // Print energy
            double ke = solver.GetKineticEnergy();
            std::cout << "  Kinetic Energy: " << std::scientific << ke << " J" << std::endl;
        });

        // Solve
        solver.Solve();

        // Write final results
        solver.WriteResults(config.output_file);

        std::cout << "\n========================================" << std::endl;
        std::cout << "           Simulation Complete!         " << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
