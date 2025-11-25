import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_PATHS = [
    os.path.join(ROOT, "build"),
    os.path.join(ROOT, "build", "src", "python"),
    os.path.join(ROOT, "build", "src", "python", "Release"),
    os.path.join(ROOT, "build", "src", "python", "Debug"),
]
for path in PYTHON_PATHS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import femml


def main():
    importer = femml.AbaqusImporter()
    mesh_path = os.path.join(ROOT, "examples", "simple_cube", "cube.inp")
    mesh = importer.import_mesh(mesh_path)

    material = femml.create_aluminum()
    solver = femml.GPUExplicitSolver(mesh)
    solver.set_material(material)
    solver.set_device_id(0)

    bc = femml.BoundaryCondition()
    bc.type = femml.BCType.FIXED
    bc.nodes = [1, 2, 5, 6]
    bc.component = -1
    solver.add_boundary_condition(bc)

    for node_id in [3, 4, 7, 8]:
        load = femml.Load()
        load.type = femml.LoadType.FORCE
        load.nodes = [node_id]
        load.component = 1
        load.value = 5e5
        solver.add_load(load)

    params = femml.SolverParams()
    params.time_step = 5e-8
    params.num_steps = 500
    params.damping = 0.02
    params.auto_time_step = False
    solver.set_parameters(params)

    solver.initialize()
    solver.solve()

    disp = solver.get_node_displacement(8)
    print(f"Node 8 displacement (m): {disp}")
    print(f"GPU compute time (s): {solver.get_gpu_compute_time():.6f}")
    print(f"GPU memory usage (MB): {solver.get_gpu_memory_usage() / (1024 * 1024):.2f}")


if __name__ == "__main__":
    main()
