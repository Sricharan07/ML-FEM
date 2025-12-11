// Simple cube example in C++
#include "mesh/AbaqusImporter.hpp"
#include "material/LinearElastic.hpp"
#include "solver/ExplicitSolver.hpp"

#include <algorithm>
#include <cmath>
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

        // Boundary conditions
        const std::vector<int> left_nodes = {1, 4, 7, 10, 13, 16};
        const std::vector<int> right_nodes = {3, 6, 9, 12, 15, 18};
        const double target_disp = 5.43e-6;  // meters
        const double ramp_time = 0.002;      // seconds

        BoundaryCondition bc;
        bc.type = BCType::FIXED;
        bc.nodes = left_nodes;
        bc.component = -1;
        solver.AddBoundaryCondition(bc);

        BoundaryCondition disp_bc;
        disp_bc.type = BCType::DISPLACEMENT;
        disp_bc.nodes = right_nodes;
        disp_bc.component = 0;  // X direction
        disp_bc.value = target_disp;
        disp_bc.ramp_time = ramp_time;
        solver.AddBoundaryCondition(disp_bc);

        // Parameters (2 ms total, 0.5 us time step)
        SolverParams params;
        params.time_step = 5e-7;
        params.num_steps = static_cast<int>(std::ceil(ramp_time / params.time_step));
        params.output_interval = std::max(1, params.num_steps / 20);
        params.damping = 0.0;
        params.auto_time_step = false;
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
