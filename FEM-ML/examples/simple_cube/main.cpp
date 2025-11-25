// Simple cube example in C++
#include "mesh/AbaqusImporter.hpp"
#include "material/LinearElastic.hpp"
#include "solver/ExplicitSolver.hpp"

#include <iostream>

using namespace femml;

int main() {
    std::cout << "========================================\n";
    std::cout << "  FEM-ML: Simple Cube Example (C++)\n";
    std::cout << "========================================\n\n";

    try {
        // Import mesh
        std::cout << "1. Importing mesh...\n";
        AbaqusImporter importer;
        auto mesh = importer.Import("cube.inp");

        // Create material (aluminum)
        std::cout << "\n2. Creating material...\n";
        auto material = CreateAluminum();
        std::cout << "   Material: " << material->GetName() << "\n";

        // Create solver
        std::cout << "\n3. Setting up solver...\n";
        ExplicitSolver solver(mesh);
        solver.SetMaterial(material);

        // Boundary conditions (fix bottom)
        BoundaryCondition bc;
        bc.type = BCType::FIXED;
        bc.nodes = {1, 2, 5, 6};
        bc.component = -1;
        solver.AddBoundaryCondition(bc);

        // Load (pull top)
        for (int node : {3, 4, 7, 8}) {
            Load load;
            load.type = LoadType::FORCE;
            load.nodes = {node};
            load.component = 1;  // Y
            load.value = 5e5;
            solver.AddLoad(load);
        }

        // Parameters
        SolverParams params;
        params.time_step = 5e-8;
        params.num_steps = 2000;
        params.output_interval = 50;
        params.auto_time_step = true;
        solver.SetParameters(params);

        // Initialize and solve
        std::cout << "\n4. Running analysis...\n";
        solver.Initialize();
        solver.Solve();

        // Write results
        std::cout << "\n5. Writing results...\n";
        solver.WriteResults("results_cpp.csv");

        std::cout << "\n========================================\n";
        std::cout << "           Analysis Complete!\n";
        std::cout << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
