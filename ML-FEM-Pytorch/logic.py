"""
PyTorch-based Explicit FEM Solver

This module provides a fully vectorized explicit dynamics solver with:
- Mass scaling for adaptive time stepping
- Hourglass stabilization  
- Bulk viscosity damping
- GPU acceleration via PyTorch tensors
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict

__all__ = [
    'PyTorchExplicitSolver',
    'SimulationConfig',
    'MaterialProperties', 
    'BoundaryCondition',
    'SimulationResults',
    'create_structured_mesh',
    'run_quick_cube_demo',
]


@dataclass
class SimulationConfig:
    """Configuration parameters for the explicit solver."""
    total_time: float = 0.01  # seconds
    target_time_step: float = 5e-6  # seconds
    safety_factor: float = 0.5  # CFL safety factor
    max_mass_scale: float = 10.0  # maximum density multiplier per scaling operation
    print_mass_scaling: bool = False
    bulk_alpha: float = 0.06  # bulk viscosity linear coefficient
    bulk_beta: float = 1.2  # bulk viscosity quadratic coefficient
    rayleigh_alpha: float = 5e-4  # Rayleigh damping coefficient
    hourglass_gamma: float = 0.05  # hourglass stiffness parameter


@dataclass
class MaterialProperties:
    """Isotropic linear elastic material properties."""
    density: float  # tonne/mm^3
    youngs_modulus: float  # MPa
    poisson_ratio: float


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    name: str
    node_indices: List[int]  # 0-based node indices
    components: Tuple[int, ...]  # (-1,) for all, or (0,), (1,), (2,) for X,Y,Z
    bc_type: str  # 'fixed' or 'displacement'
    value: float = 0.0  # mm for displacement
    ramp_time: float = 0.0  # seconds


@dataclass
class SimulationResults:
    """Container for simulation results."""
    displacements: torch.Tensor  # (ndof,) final displacements
    velocities: torch.Tensor  # (ndof,) final velocities
    element_stress: torch.Tensor  # (n_elems, 6) final stresses
    kinetic_energy: List[float] = field(default_factory=list)
    internal_energy: List[float] = field(default_factory=list)
    time_history: List[float] = field(default_factory=list)


class PyTorchExplicitSolver:
    """
    Fully vectorized explicit FEM solver using PyTorch.
    
    Features:
    - Lumped mass matrix
    - Adaptive time stepping with mass scaling
    - Hourglass stabilization
    - Bulk viscosity damping
    - GPU acceleration
    """
    
    def __init__(
        self,
        coords: torch.Tensor,  # (n_nodes, 3)
        elements: torch.Tensor,  # (n_elems, 8) for C3D8/C3D8R
        material: MaterialProperties,
        config: SimulationConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Move to device
        self.coords = coords.to(device=self.device, dtype=self.dtype)
        self.elements = elements.to(device=self.device, dtype=torch.long)
        
        self.material = material
        self.config = config
        
        self.n_nodes = self.coords.shape[0]
        self.n_elems = self.elements.shape[0]
        self.ndof = self.n_nodes * 3
        
        # Build constitutive matrix
        self.C = self._build_constitutive_matrix()
        
        # Precompute geometry
        self._precompute_geometry()
        
        # Initialize per-element density (can be modified by mass scaling)
        self.rho_e = torch.full((self.n_elems,), float(material.density), dtype=self.dtype, device=self.device)
        
        # Build masses
        self.node_mass, self.M_diag, self.M_inv_vec = self._build_lumped_masses()
        
    def _build_constitutive_matrix(self) -> torch.Tensor:
        """Build 6x6 Voigt elasticity tensor."""
        E = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        C = torch.tensor([
            [lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam+2*mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu]
        ], dtype=self.dtype, device=self.device)
        return C
    
    def _precompute_geometry(self):
        """Precompute element geometry and B-matrices at element centers."""
        el_coords_all = self.coords[self.elements]  # (n_elems, 8, 3)
        
        # Shape function derivatives at center (xi=eta=zeta=0)
        dN_dxi = torch.tensor([
            [-0.125, -0.125, -0.125],
            [ 0.125, -0.125, -0.125],
            [ 0.125,  0.125, -0.125],
            [-0.125,  0.125, -0.125],
            [-0.125, -0.125,  0.125],
            [ 0.125, -0.125,  0.125],
            [ 0.125,  0.125,  0.125],
            [-0.125,  0.125,  0.125]
        ], dtype=self.dtype, device=self.device)  # (8, 3)
        
        # Jacobian: J = dN/dxi^T @ coords
        J_all = torch.einsum('ij,eik->ejk', dN_dxi, el_coords_all)  # (n_elems, 3, 3)
        self.detJ_all = torch.linalg.det(J_all)  # (n_elems,)
        invJ_all = torch.linalg.inv(J_all)  # (n_elems, 3, 3)
        
        # Physical derivatives: dN/dX = dN/dxi @ inv(J)
        dN_dX_all = torch.einsum('ij,ejk->eik', dN_dxi, invJ_all)  # (n_elems, 8, 3)
        
        # Build B-matrix (strain-displacement): (n_elems, 6, 24)
        self.B_all = torch.zeros((self.n_elems, 6, 24), dtype=self.dtype, device=self.device)
        for i in range(8):
            self.B_all[:, 0, 3*i]   = dN_dX_all[:, i, 0]  # ∂Ni/∂x
            self.B_all[:, 1, 3*i+1] = dN_dX_all[:, i, 1]  # ∂Ni/∂y
            self.B_all[:, 2, 3*i+2] = dN_dX_all[:, i, 2]  # ∂Ni/∂z
            self.B_all[:, 3, 3*i]   = dN_dX_all[:, i, 1]  # γxy part 1
            self.B_all[:, 3, 3*i+1] = dN_dX_all[:, i, 0]  # γxy part 2
            self.B_all[:, 4, 3*i+1] = dN_dX_all[:, i, 2]  # γyz part 1
            self.B_all[:, 4, 3*i+2] = dN_dX_all[:, i, 1]  # γyz part 2
            self.B_all[:, 5, 3*i]   = dN_dX_all[:, i, 2]  # γxz part 1
            self.B_all[:, 5, 3*i+2] = dN_dX_all[:, i, 0]  # γxz part 2
        
        self.B_t = self.B_all.transpose(1, 2)  # (n_elems, 24, 6)
        
        # Element DOF indices
        node_base = self.elements * 3  # (n_elems, 8)
        offsets = torch.tensor([0, 1, 2], dtype=torch.long, device=self.device)
        self.elem_dof_idx = (node_base.unsqueeze(2) + offsets.view(1, 1, 3)).reshape(self.n_elems, 24)
        self.elem_dof_idx_flat = self.elem_dof_idx.reshape(-1).long()
        
        # Characteristic element length (min edge length)
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        edge_lengths = []
        for (a, b) in edges:
            vec = el_coords_all[:, a, :] - el_coords_all[:, b, :]
            l = torch.linalg.norm(vec, dim=1)
            edge_lengths.append(l)
        self.h_e = torch.stack(edge_lengths, dim=1).min(dim=1)[0]  # (n_elems,)
    
    def _build_lumped_masses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build lumped mass matrix from current element densities."""
        vol = self.detJ_all.abs()  # element volumes
        mass_el = self.rho_e * vol  # (n_elems,)
        
        # Distribute to nodes (1/8 per node for hex8)
        node_idx_flat = self.elements.reshape(-1)  # (n_elems*8,)
        mass_per_node_flat = (mass_el.repeat_interleave(8) / 8.0).reshape(-1)
        
        node_mass = torch.zeros(self.n_nodes, dtype=self.dtype, device=self.device)
        node_mass = node_mass.scatter_add(0, node_idx_flat, mass_per_node_flat)
        
        M_diag = node_mass.repeat_interleave(3)
        M_diag = torch.clamp(M_diag, min=1e-18)
        M_inv_vec = 1.0 / M_diag
        
        return node_mass, M_diag, M_inv_vec
    
    def run(
        self,
        boundary_conditions: List[BoundaryCondition],
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> SimulationResults:
        """
        Run the explicit time integration.
        
        Args:
            boundary_conditions: List of boundary conditions to apply
            progress_callback: Optional callback function for progress updates
        
        Returns:
            SimulationResults containing final state and time history
        """
        # Initialize state
        u = torch.zeros(self.ndof, dtype=self.dtype, device=self.device)
        v = torch.zeros(self.ndof, dtype=self.dtype, device=self.device)
        stress_e = torch.zeros((self.n_elems, 6), dtype=self.dtype, device=self.device)
        u_prev_el = torch.zeros((self.n_elems, 24), dtype=self.dtype, device=self.device)

        # helper to scrub NaN/Inf explosions that can happen with aggressive BCs
        def safe(t: torch.Tensor, clip: float = 1e12) -> torch.Tensor:
            return torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
        
        # Prepare BC DOF lists
        fixed_dofs = []
        disp_dofs = []
        disp_values = []
        disp_ramp_times = []
        
        for bc in boundary_conditions:
            for node_idx in bc.node_indices:
                if bc.components == (-1,):  # All DOFs
                    dof_list = [node_idx * 3 + i for i in range(3)]
                else:
                    dof_list = [node_idx * 3 + c for c in bc.components]
                
                if bc.bc_type == 'fixed':
                    fixed_dofs.extend(dof_list)
                elif bc.bc_type == 'displacement':
                    disp_dofs.extend(dof_list)
                    disp_values.extend([bc.value] * len(dof_list))
                    disp_ramp_times.extend([bc.ramp_time] * len(dof_list))
        
        fixed_dofs = torch.tensor(fixed_dofs, dtype=torch.long, device=self.device)
        disp_dofs = torch.tensor(disp_dofs, dtype=torch.long, device=self.device)
        disp_values = torch.tensor(disp_values, dtype=self.dtype, device=self.device)
        disp_ramp_times = torch.tensor(disp_ramp_times, dtype=self.dtype, device=self.device)
        
        # Time integration
        t = 0.0
        step = 0
        KE_list, IE_list, time_list = [], [], []
        
        print(f"Starting simulation: total_time={self.config.total_time}s, target_dt={self.config.target_time_step}s")
        
        while t < self.config.total_time:
            # Compute stable time step
            rho_e_clamped = torch.clamp(self.rho_e, min=1e-20)
            c_e = torch.sqrt(self.material.youngs_modulus / rho_e_clamped)
            dt_e = self.h_e / c_e
            dt_e_min = float(torch.min(dt_e).item())
            
            # Mass scaling if needed
            if dt_e_min < self.config.target_time_step:
                e_star = int(torch.argmin(dt_e).item())
                required_factor = (self.config.target_time_step / dt_e[e_star]) ** 2
                factor = min(float(required_factor), self.config.max_mass_scale)
                if factor > 1.0:
                    self.rho_e[e_star] *= factor
                    self.node_mass, self.M_diag, self.M_inv_vec = self._build_lumped_masses()
                    if self.config.print_mass_scaling:
                        print(f"  [mass-scale] step={step}, elem={e_star}, factor={factor:.2f}")
                    # Recompute dt
                    rho_e_clamped = torch.clamp(self.rho_e, min=1e-20)
                    c_e = torch.sqrt(self.material.youngs_modulus / rho_e_clamped)
                    dt_e = self.h_e / c_e
                    dt_e_min = float(torch.min(dt_e).item())
            
            dt = self.config.safety_factor * dt_e_min
            if t + dt > self.config.total_time:
                dt = self.config.total_time - t
            
            # Apply prescribed displacements (ramp)
            if len(disp_dofs) > 0:
                ramp_factor = torch.where(
                    disp_ramp_times > 0,
                    torch.clamp(torch.tensor(t, dtype=self.dtype, device=self.device) / disp_ramp_times, 0.0, 1.0),
                    torch.ones_like(disp_ramp_times)
                )
                u[disp_dofs] = disp_values * ramp_factor
                v[disp_dofs] = 0.0
            
            # Compute element displacements
            u_nodes = u.view(self.n_nodes, 3)
            u_el = u_nodes[self.elements].reshape(self.n_elems, 24)
            du_el = u_el - u_prev_el
            u_prev_el = u_el.clone()
            
            # Incremental strain
            strain_inc = torch.einsum('eij,ej->ei', self.B_all, du_el)  # (n_elems, 6)
            strain_inc = safe(strain_inc)
            
            # Bulk viscosity
            eps_dot_vol = (strain_inc[:, 0] + strain_inc[:, 1] + strain_inc[:, 2]) / dt
            c_bulk = torch.sqrt(self.material.youngs_modulus / torch.tensor(self.material.density, dtype=self.dtype, device=self.device))
            bulk_stress = (
                self.config.bulk_alpha * self.rho_e.mean() * c_bulk * self.h_e.min() * eps_dot_vol +
                self.config.bulk_beta * self.rho_e.mean() * (self.h_e.min() ** 2) * (eps_dot_vol ** 2)
            )
            bulk_stress = safe(bulk_stress)
            strain_inc[:, 0:3] += (bulk_stress / self.material.youngs_modulus).unsqueeze(1)
            
            # Stress update
            delta_stress = torch.einsum('ij,ej->ei', self.C, strain_inc)
            delta_stress = safe(delta_stress)
            stress_e = stress_e + delta_stress
            stress_e = safe(stress_e)
            
            # Internal forces
            f_el = torch.bmm(self.B_t, stress_e.unsqueeze(2)).squeeze(2) * self.detJ_all[:, None]
            f_el = safe(f_el)
            
            # Hourglass stabilization
            du_el_nodes = du_el.reshape(self.n_elems, 8, 3)
            mean_du = du_el_nodes.mean(dim=1, keepdim=True)
            hg_mode = du_el_nodes - mean_du
            hg_force_nodes = -self.config.hourglass_gamma * hg_mode * self.detJ_all[:, None, None]
            f_el = f_el + hg_force_nodes.reshape(self.n_elems, 24)
            f_el = safe(f_el)
            
            # Assemble
            f_int = torch.zeros(self.ndof, dtype=self.dtype, device=self.device)
            f_int = f_int.scatter_add(0, self.elem_dof_idx_flat, f_el.reshape(-1))
            f_int = safe(f_int)
            
            # Acceleration
            a = -f_int * self.M_inv_vec
            a = safe(a, clip=1e9)
            
            # Velocity update with Rayleigh damping
            v = v + a * dt
            v = v * (1.0 - self.config.rayleigh_alpha * dt)
            v = safe(v, clip=1e6)
            
            # Displacement update
            u = u + v * dt
            u = safe(u, clip=1e6)
            
            # Reapply BCs
            if len(fixed_dofs) > 0:
                u[fixed_dofs] = 0.0
                v[fixed_dofs] = 0.0
            if len(disp_dofs) > 0:
                ramp_factor = torch.where(
                    disp_ramp_times > 0,
                    torch.clamp(torch.tensor(t + dt, dtype=self.dtype, device=self.device) / disp_ramp_times, 0.0, 1.0),
                    torch.ones_like(disp_ramp_times)
                )
                u[disp_dofs] = disp_values * ramp_factor
                v[disp_dofs] = 0.0
            
            # Energies
            KE = 0.5 * torch.sum(self.M_diag * v ** 2)
            IE = 0.5 * torch.sum(stress_e ** 2) / self.material.youngs_modulus * self.detJ_all.mean()
            
            KE_list.append(float(KE.item()))
            IE_list.append(float(IE.item()))
            time_list.append(float(t + dt))
            
            t += dt
            step += 1
            
            # Progress callback (reduced frequency for better GUI performance)
            if progress_callback and step % 20 == 0:
                progress_callback({
                    'step': step,
                    'time': t,
                    'dt': dt,
                    'ke': float(KE.item()),
                    'ie': float(IE.item()),
                    'displacements': u.view(self.n_nodes, 3),
                    'element_stress': stress_e,
                })
            
            # Console output (less frequent)
            if step % 100 == 0:
                print(f"  step={step}, t={t:.6e}s, dt={dt:.3e}s, KE={KE:.3e}, IE={IE:.3e}")
        
        print(f"Simulation complete: {step} steps")
        
        return SimulationResults(
            displacements=u,
            velocities=v,
            element_stress=stress_e,
            kinetic_energy=KE_list,
            internal_energy=IE_list,
            time_history=time_list
        )


def create_structured_mesh(
    nx: int, ny: int, nz: int,
    lengths: Tuple[float, float, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a structured hexahedral mesh.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        lengths: Dimensions (Lx, Ly, Lz) in mm
    
    Returns:
        coords: (n_nodes, 3) node coordinates
        elements: (n_elems, 8) element connectivity
    """
    Lx, Ly, Lz = lengths
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    
    coords_list = [
        [i*dx, j*dy, k*dz]
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]
    coords = torch.tensor(coords_list, dtype=torch.float64)
    
    elements_list = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = k*(nx+1)*(ny+1) + j*(nx+1) + i
                n1 = n0 + 1
                n2 = n0 + (nx+1) + 1
                n3 = n0 + (nx+1)
                n4 = n0 + (nx+1)*(ny+1)
                n5 = n4 + 1
                n6 = n4 + (nx+1) + 1
                n7 = n4 + (nx+1)
                elements_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    elements = torch.tensor(elements_list, dtype=torch.long)
    
    return coords, elements


def _plot_quick_results(coords: torch.Tensor, elements: torch.Tensor, result: SimulationResults) -> None:
    """Lightweight plotting for the quick cube demo."""
    coords_np = coords.cpu().numpy()
    elements_np = elements.cpu().numpy()
    stress_np = result.element_stress.cpu().numpy()
    disp_nodes = result.displacements.view(-1, 3).cpu().numpy()
    
    # Element centers for stress plots
    centers = coords_np[elements_np].mean(axis=1)
    sigma_xx = stress_np[:, 0]
    sigma_yy = stress_np[:, 1]
    sigma_zz = stress_np[:, 2]
    sigma_xy = stress_np[:, 3]
    sigma_yz = stress_np[:, 4]
    sigma_xz = stress_np[:, 5]
    sigma_vm = np.sqrt(
        0.5 * (
            (sigma_xx - sigma_yy) ** 2 +
            (sigma_yy - sigma_zz) ** 2 +
            (sigma_zz - sigma_xx) ** 2 +
            6.0 * (sigma_xy ** 2 + sigma_yz ** 2 + sigma_xz ** 2)
        )
    )
    
    disp_mag = np.linalg.norm(disp_nodes, axis=1)
    
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    
    sc0 = axs[0].scatter(centers[:, 0], centers[:, 1], c=sigma_xx, cmap='viridis', s=160, edgecolors='k')
    axs[0].set_title('sigma_xx [MPa]')
    axs[0].set_xlabel('x [mm]')
    axs[0].set_ylabel('y [mm]')
    fig.colorbar(sc0, ax=axs[0], fraction=0.046)
    
    sc1 = axs[1].scatter(centers[:, 0], centers[:, 1], c=sigma_vm, cmap='plasma', s=160, edgecolors='k')
    axs[1].set_title('von Mises [MPa]')
    axs[1].set_xlabel('x [mm]')
    axs[1].set_ylabel('y [mm]')
    fig.colorbar(sc1, ax=axs[1], fraction=0.046)
    
    sc2 = axs[2].scatter(coords_np[:, 0], coords_np[:, 1], c=disp_mag, cmap='coolwarm', s=100, edgecolors='k')
    axs[2].set_title('|u| [mm] at nodes')
    axs[2].set_xlabel('x [mm]')
    axs[2].set_ylabel('y [mm]')
    fig.colorbar(sc2, ax=axs[2], fraction=0.046)
    
    plt.tight_layout()
    plt.show()


def run_quick_cube_demo() -> None:
    """
    Quick CPU-friendly demo: 1 mm cube, left face clamped, right face pulled 0.75 mm.
    Designed to finish in a few seconds and produce stress plots.
    """
    # Geometry: small cube with 2x2x2 elements (8 elements total)
    lengths = (1.0, 1.0, 1.0)  # mm
    nx = ny = nz = 2
    coords, elements = create_structured_mesh(nx, ny, nz, lengths)
    
    # Aluminum properties (tonne/mm^3, MPa)
    material = MaterialProperties(
        density=2.7e-9,
        youngs_modulus=70000.0,
        poisson_ratio=0.32,
    )
    
    # Short run time for quick CPU execution
    config = SimulationConfig(
        total_time=0.003,          # seconds
        target_time_step=5e-6,     # seconds (mass scaling will hit this quickly)
        safety_factor=0.5,
        max_mass_scale=1e4,        # allow aggressive scaling to reduce steps
        print_mass_scaling=False,
        bulk_alpha=0.06,
        bulk_beta=1.2,
        rayleigh_alpha=5e-4,
        hourglass_gamma=0.05,
    )
    
    # Boundary conditions
    x_vals = coords[:, 0].numpy()
    left_nodes = np.where(np.isclose(x_vals, 0.0))[0].tolist()
    right_nodes = np.where(np.isclose(x_vals, lengths[0]))[0].tolist()
    
    bcs = [
        BoundaryCondition(name="Clamp-Left", node_indices=left_nodes, components=(-1,), bc_type="fixed"),
        BoundaryCondition(
            name="Pull-Right",
            node_indices=right_nodes,
            components=(0,),
            bc_type="displacement",
            value=0.75,                  # mm
            ramp_time=config.total_time  # ramp over the short step to avoid impulse
        ),
    ]
    
    solver = PyTorchExplicitSolver(
        coords,
        elements,
        material=material,
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    
    result = solver.run(bcs)
    
    u_nodes = result.displacements.view(-1, 3).cpu().numpy()
    max_disp = np.linalg.norm(u_nodes, axis=1).max()
    print(f"Max nodal displacement: {max_disp:.4f} mm")
    
    _plot_quick_results(coords, elements, result)


if __name__ == "__main__":
    run_quick_cube_demo()
