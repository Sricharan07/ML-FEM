import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import math
import time as pytime
from matplotlib.collections import LineCollection
import copy

# ----------------------------
# Device & dtype
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ----------------------------
# Material properties (Aluminum)
# ----------------------------
rho0 = 2.7e-9             # base density (tonne / mm^3)
E = 70000.0               # MPa
nu = 0.32                 # Poisson's ratio

E_t = torch.tensor(E, dtype=DTYPE, device=device)

# Constitutive matrix (Voigt 6x6)
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
C = torch.tensor([
    [lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
    [lam, lam+2*mu, lam, 0.0, 0.0, 0.0],
    [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, mu, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, mu, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, mu]
], dtype=DTYPE, device=device)

# ----------------------------
# Mesh (structured hexahedral)
# ----------------------------
nx, ny, nz = 100, 30, 1               # elements in x,y,z directions
Lx, Ly, Lz = 138.0, 38.0, 1.0         # mm (solid dims)
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

coords_list = [[i*dx, j*dy, k*dz] for k in range(nz+1) for j in range(ny+1) for i in range(nx+1)]
coords = torch.tensor(coords_list, dtype=DTYPE, device=device)  # (n_nodes,3)
n_nodes = coords.shape[0]

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
            elements_list.append([n0,n1,n2,n3,n4,n5,n6,n7])
elements = torch.tensor(elements_list, dtype=torch.long, device=device)  # (n_elems,8)
n_elems = elements.shape[0]
ndof = n_nodes * 3

# ----------------------------
# Per-element density array (we will update some entries during mass-scaling)
# ----------------------------
rho_e = torch.full((n_elems,), float(rho0), dtype=DTYPE, device=device)  # tonne/mm^3 per element

# ----------------------------
# Element DOF mapping (n_elems,24) and flattened index for scatter_add
# ----------------------------
node_base = elements * 3                                   # (n_elems,8)
offsets = torch.tensor([0,1,2], dtype=torch.long, device=device)
elem_dof_idx = (node_base.unsqueeze(2) + offsets.view(1,1,3)).reshape(n_elems,24)
elem_dof_idx_flat = elem_dof_idx.reshape(-1).long()

# ----------------------------
# Precompute element coordinates and natural derivative matrix (at center)
# ----------------------------
el_coords_all = coords[elements]    # (n_elems,8,3)
xi = eta = zeta = 0.0
dN_dxi = torch.tensor([
    [-(1-eta)*(1-zeta)/8, -(1-xi)*(1-zeta)/8, -(1-xi)*(1-eta)/8],
    [(1-eta)*(1-zeta)/8, -(1+xi)*(1-zeta)/8, -(1+xi)*(1-eta)/8],
    [(1+eta)*(1-zeta)/8, (1+xi)*(1-zeta)/8, -(1+xi)*(1+eta)/8],
    [-(1+eta)*(1-zeta)/8, (1-xi)*(1-zeta)/8, -(1-xi)*(1+eta)/8],
    [-(1-eta)*(1+zeta)/8, -(1-xi)*(1+zeta)/8, (1-xi)*(1-eta)/8],
    [(1-eta)*(1+zeta)/8, -(1+xi)*(1+zeta)/8, (1+xi)*(1-eta)/8],
    [(1+eta)*(1+zeta)/8, (1+xi)*(1+zeta)/8, (1+xi)*(1+eta)/8],
    [-(1+eta)*(1+zeta)/8, (1-xi)*(1+zeta)/8, (1-xi)*(1+eta)/8]
], dtype=DTYPE, device=device)  # (8,3)

# Compute reference jacobians (for detJ of shape function mapping from xi->X)
J_all = torch.einsum('ij,eik->ejk', dN_dxi, el_coords_all)  # (n_elems,3,3)
detJ_all = torch.linalg.det(J_all)                          # (n_elems,)
# compute invJ for center (used if needed)
invJ_all = torch.linalg.inv(J_all)
# physical shape function derivatives dN/dX at center per element: (n_elems,8,3)
dN_dX_all = torch.einsum('ij,ejk->eik', dN_dxi, invJ_all)

# Build B_all: (n_elems,6,24)
B_all = torch.zeros((n_elems,6,24), dtype=DTYPE, device=device)
for i in range(8):
    B_all[:,0,3*i]   = dN_dX_all[:,i,0]
    B_all[:,1,3*i+1] = dN_dX_all[:,i,1]
    B_all[:,2,3*i+2] = dN_dX_all[:,i,2]
    B_all[:,3,3*i]   = dN_dX_all[:,i,1]
    B_all[:,3,3*i+1] = dN_dX_all[:,i,0]
    B_all[:,4,3*i+1] = dN_dX_all[:,i,2]
    B_all[:,4,3*i+2] = dN_dX_all[:,i,1]
    B_all[:,5,3*i]   = dN_dX_all[:,i,2]
    B_all[:,5,3*i+2] = dN_dX_all[:,i,0]

# ----------------------------
# Define semi-circular notch: center (x=69, y=0), radius 3.175 mm, semi-circle opens upward
# Mark elements whose element-centroid lies inside the semi-circle as invalid (hole)
# ----------------------------
hole_center_x = 69.0
hole_center_y = 0.0
hole_R = 3.175

# compute element centers (move to CPU for geometric tests)
elem_centers = el_coords_all.mean(dim=1).cpu().numpy()  # (n_elems,3)
elem_x = elem_centers[:,0]
elem_y = elem_centers[:,1]

inside = ((elem_x - hole_center_x)**2 + (elem_y - hole_center_y)**2) <= (hole_R**2)
inside = np.logical_and(inside, elem_y >= 0.0)  # semi-circle opens upward

hole_mask = torch.tensor(inside, dtype=torch.bool, device=device)  # True for elements INSIDE hole
valid_elems = ~hole_mask  # True for valid elements

# zero out density for hole elements so they contribute nothing
rho_e[hole_mask] = 0.0

# ----------------------------
# Characteristic element length: compute min edge length per element
# ----------------------------
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
el_node_coords = el_coords_all.cpu().numpy()  # bring to CPU for length calc
edge_lengths = []
for (a,b) in edges:
    vec = el_node_coords[:,a,:] - el_node_coords[:,b,:]   # (n_elems,3)
    l = np.linalg.norm(vec, axis=1)                      # (n_elems,)
    edge_lengths.append(l)
edge_lengths = np.stack(edge_lengths, axis=1)           # (n_elems,12)
h_e = np.min(edge_lengths, axis=1)                      # (n_elems,) characteristic min edge length
h_e = torch.tensor(h_e, dtype=DTYPE, device=device)

# fallback dx_min scalar
dx_min = float(min(dx, dy, dz))

# ----------------------------
# Lumped mass initialization (function)
# ----------------------------
vol = dx * dy * dz   # element volume (mm^3), uniform for structured mesh
def build_lumped_masses(rho_e_local):
    """
    Given per-element density rho_e_local (n_elems,),
    compute node_mass (n_nodes,) and global M_diag (ndof,) and M_inv_vec.
    """
    mass_el_vec = rho_e_local * vol  # (n_elems,)
    node_idx_flat = elements.reshape(-1)                     # (n_elems*8,)
    mass_per_node_flat = (mass_el_vec.repeat_interleave(8) / 8.0).reshape(-1)  # (n_elems*8,)
    node_mass = torch.zeros(n_nodes, dtype=DTYPE, device=device)
    node_mass = node_mass.scatter_add(0, node_idx_flat, mass_per_node_flat)
    M_diag_local = node_mass.repeat_interleave(3)
    M_diag_local[M_diag_local == 0.0] = 1e-18
    M_inv_vec_local = 1.0 / M_diag_local
    return node_mass, M_diag_local, M_inv_vec_local

node_mass, M_diag, M_inv_vec = build_lumped_masses(rho_e)

# ----------------------------
# Boundary DOFs: left fixed, right u_x prescribed
# ----------------------------
x_vals = coords[:,0]
left_mask_nodes = torch.isclose(x_vals, torch.tensor(0.0, dtype=DTYPE, device=device), atol=1e-8)
left_nodes = torch.nonzero(left_mask_nodes).view(-1)
left_dofs = (left_nodes.unsqueeze(1)*3 + torch.tensor([0,1,2], device=device)).reshape(-1).long()

right_mask_nodes = torch.isclose(x_vals, torch.tensor(Lx, dtype=DTYPE, device=device), atol=1e-8)
right_nodes = torch.nonzero(right_mask_nodes).view(-1)
right_dofs = (right_nodes*3 + 0).long()  # x DOFs

# ----------------------------
# Initialize fields
# ----------------------------
u = torch.zeros(ndof, dtype=DTYPE, device=device)
v = torch.zeros(ndof, dtype=DTYPE, device=device)
stress_e = torch.zeros((n_elems,6), dtype=DTYPE, device=device)
u_prev_el = torch.zeros((n_elems,24), dtype=DTYPE, device=device)

# ensure hole elements stress stays zero and previous state zero
stress_e[hole_mask,:] = 0.0
u_prev_el[hole_mask, :] = 0.0

# ----------------------------
# Bulk viscosity, damping, hourglass
# ----------------------------
alpha = 0.06
beta  = 1.2
rayleigh_alpha = 5e-4
hg_gamma = 0.05

# ----------------------------
# Mass-scaling parameters (tunable)
# ----------------------------
dt_target = 1e-6             # target timestep (s) we want to reach via scaling (choose as you like)
max_scale_per_step = 10.0    # maximum multiplicative density increase allowed in one mass-scaling operation
safety_factor = 0.5          # use dt = safety_factor * dt_elem_min (typical 0.5)
print_mass_scaling = True

# ----------------------------
# Loading: prescribe u_x_total over total_time seconds
# ux = 0.75 mm over total_time = 0.3 s
# ----------------------------
ux_total = 0.75      # mm (final prescribed x displacement on right face)
total_time = 0.3     # seconds
t = 0.0
step = 0

# For plotting quantities
KE_list = []
IE_list = []
time_list = []
maxux_list = []

# ----------------------------
# Prepare some tensors reused in loop
# ----------------------------
B_t = B_all.transpose(1,2)  # (n_elems,24,6)
elem_dof_idx_flat_long = elem_dof_idx_flat  # alias
detJ_all_local = detJ_all  # alias

# ----------------------------
# MAIN TIME LOOP (adaptive dt, mass-scaling)
# ----------------------------
start_time_wall = pytime.time()
last_print_time = start_time_wall
print_every_seconds = max(1.0, total_time / 20.0)  # print every ~fraction of total time

# Precompute hole mask as numpy for dt array masking later
hole_mask_cpu = hole_mask.cpu().numpy()

while t < total_time:
    # --- compute per-element wave speed and stable dt
    rho_e_clamped = torch.clamp(rho_e, min=1e-20)  # avoid zeros
    c_e = torch.sqrt(E_t / rho_e_clamped)          # (n_elems,) mm/s
    dt_e = h_e / c_e                                # (n_elems,) seconds

    # ignore hole elements when computing min dt (they have rho=0 and produce huge c_e/infs)
    dt_e_cpu = dt_e.cpu().numpy()
    dt_e_cpu[hole_mask_cpu] = np.inf
    dt_e_min = float(np.min(dt_e_cpu))

    # --- apply Abaqus-style mass scaling if critical element is below target
    if dt_e_min < dt_target:
        # find critical element(s) among valid elements
        valid_dt_e = torch.tensor(dt_e_cpu, dtype=DTYPE, device=device)
        e_star = int(torch.argmin(valid_dt_e).item())
        dt_star = float(dt_e[e_star].item())
        required_factor = (dt_target / dt_star)**2
        factor = float(min(required_factor, max_scale_per_step))
        if factor > 1.0:
            old_rho = float(rho_e[e_star].item())
            rho_e[e_star] = rho_e[e_star] * factor
            node_mass, M_diag, M_inv_vec = build_lumped_masses(rho_e)
            if print_mass_scaling:
                print(f"[mass-scaling] step {step} t={t:.6e}s: element {e_star} dt={dt_star:.3e}s -> "
                      f"scale {factor:.3f}, rho_old={old_rho:.3e}, rho_new={float(rho_e[e_star].item()):.3e}")
            rho_e_clamped = torch.clamp(rho_e, min=1e-20)
            c_e = torch.sqrt(E_t / rho_e_clamped)
            dt_e = h_e / c_e
            dt_e_cpu = dt_e.cpu().numpy()
            dt_e_cpu[hole_mask_cpu] = np.inf
            dt_e_min = float(np.min(dt_e_cpu))

    # --- choose current timestep (safety factor)
    dt_current = safety_factor * dt_e_min
    if t + dt_current > total_time:
        dt_current = total_time - t

    # --- ramp prescribed ux linearly over total_time
    ux_ref = ux_total * (t / total_time) if total_time > 0 else ux_total
    if t + dt_current >= total_time:
        ux_ref = ux_total

    step += 1
    # apply prescribed boundary (right face x displacement)
    u[right_dofs] = ux_ref
    v[right_dofs] = 0.0

    # --- element nodal displacements and incremental displacement per element
    u_nodes = u.view(n_nodes, 3)
    u_el = u_nodes[elements].reshape(n_elems, 24)
    du_el = u_el - u_prev_el
    u_prev_el = u_el.clone()

    # --- incremental strain (linear small strain increment using B at center)
    strain_inc = torch.einsum('eij,ej->ei', B_all, du_el)   # (n_elems,6)

    # zero strain_inc for hole elements (so they don't produce stress)
    strain_inc[hole_mask,:] = 0.0

    # --- bulk viscosity (simple)
    eps_dot_vol = (strain_inc[:,0] + strain_inc[:,1] + strain_inc[:,2]) / dt_current
    # use mean density for bulk term as before (hole elements have rho=0 and produce no strain)
    bulk_stress = alpha * rho_e.mean() * float(torch.sqrt(E_t / torch.tensor(float(rho0), dtype=DTYPE, device=device)).item()) * dx_min * eps_dot_vol \
                  + beta * rho_e.mean() * (dx_min**2) * (eps_dot_vol**2)
    if E != 0.0:
        # expand (n_elems,) to (n_elems,1) so broadcasting works
        strain_inc[:,0:3] += (bulk_stress / E).unsqueeze(1)

    # --- stress update (linear elastic incremental)
    delta_stress = torch.einsum('ij,ej->ei', C, strain_inc)  # (n_elems,6)
    # zero delta_stress for hole elements (just in case)
    delta_stress[hole_mask,:] = 0.0
    stress_e = stress_e + delta_stress
    stress_e[hole_mask,:] = 0.0  # keep hole stresses zero

    # --- internal element forces
    f_el = torch.bmm(B_t, stress_e.unsqueeze(2)).squeeze(2) * detJ_all_local[:, None]  # (n_elems,24)
    f_el[hole_mask,:] = 0.0

    # --- hourglass stabilization (simple penalty)
    du_el_nodes = du_el.reshape(n_elems,8,3)
    mean_du = du_el_nodes.mean(dim=1, keepdim=True)
    hg_mode = du_el_nodes - mean_du
    hg_force_nodes = -hg_gamma * hg_mode * detJ_all_local[:, None, None]
    hg_force_flat = hg_force_nodes.reshape(n_elems,24)
    hg_force_flat[hole_mask,:] = 0.0
    f_el = f_el + hg_force_flat

    # --- assemble global internal force
    f_int = torch.zeros(ndof, dtype=DTYPE, device=device)
    f_int = f_int.scatter_add(0, elem_dof_idx_flat_long, f_el.reshape(-1))

    # --- acceleration (f_ext assumed zero except imposed disp)
    a = -f_int * M_inv_vec   # (ndof,)

    # --- velocity and displacement update (explicit)
    v = v + a * dt_current
    v = v * (1.0 - rayleigh_alpha * dt_current)
    u = u + v * dt_current

    # reapply essential BCs
    if left_dofs.numel() > 0:
        u[left_dofs] = 0.0
        v[left_dofs] = 0.0
    u[right_dofs] = ux_ref
    v[right_dofs] = 0.0

    # --- energies and monitoring
    KE = 0.5 * torch.sum(M_diag * v**2)   # in arbitrary units
    IE = 0.5 * torch.sum(stress_e**2) / E * vol  # rough proxy
    KE_list.append(float(KE.item()))
    IE_list.append(float(IE.item()))
    time_list.append(float(t + dt_current))
    maxux = float(u[0::3].abs().max().item())
    maxux_list.append(maxux)

    # update time
    t += dt_current

    # print periodically
    wall_now = pytime.time()
    if (wall_now - last_print_time) >= print_every_seconds or t >= total_time or step == 1:
        last_print_time = wall_now
        print(f"Step {step:6d}, t={t:.6e}s, dt={dt_current:.3e}s, dt_min_elem={dt_e_min:.3e}s, "
              f"KE={KE:.3e}, IE={IE:.3e}, max|u_x|={maxux:.6e}")

# total run time
print(f"Simulation finished in {pytime.time() - start_time_wall:.1f} s wall-clock time, steps = {step}")

# ----------------------------
# Postprocess: extrapolate element center stresses to nodes (simple average)
# Only use valid elements when extrapolating so hole stays white
# ----------------------------
stress_e_cpu = stress_e.cpu().numpy()  # (n_elems,6)
elements_cpu = elements.cpu().numpy()
valid_elems_cpu = valid_elems.cpu().numpy()

sigma_nodes_np = np.zeros((n_nodes,6), dtype=float)
counts_np = np.zeros((n_nodes,), dtype=float)

for i in range(n_elems):
    if not valid_elems_cpu[i]:
        continue
    s_e = stress_e_cpu[i, :]  # (6,)
    el_nodes = elements_cpu[i]  # 8 node indices
    for ni in range(8):
        nid = int(el_nodes[ni])
        sigma_nodes_np[nid, :] += s_e * (1.0/8.0)
        counts_np[nid] += 1.0/8.0

counts_np[counts_np == 0.0] = 1.0
sigma_nodes_np = sigma_nodes_np / counts_np[:,None]

# move sigma_nodes back to tensor if needed later
sigma_nodes = torch.tensor(sigma_nodes_np, dtype=DTYPE, device=device)

# choose bottom layer (z=0) for plotting
sigma_np = sigma_nodes.cpu().numpy()
u_np = u.cpu().numpy()
coords_np = coords.cpu().numpy()
x = coords_np[:,0]; y = coords_np[:,1]; z = coords_np[:,2]
z_layer = 0.0
layer_nodes = np.where(np.isclose(z, z_layer))[0]
x_layer, y_layer = x[layer_nodes], y[layer_nodes]
sigma_xx_layer = sigma_np[layer_nodes,0]
sigma_yy_layer = sigma_np[layer_nodes,1]
sigma_xy_layer = sigma_np[layer_nodes,3]
u_x_layer = u_np[0::3][layer_nodes]
u_y_layer = u_np[1::3][layer_nodes]
u_z_layer = u_np[2::3][layer_nodes]

# ----------------------------
# Smooth onto regular grid
# ----------------------------
def smooth_grid(x_vals, y_vals, values, nx_fine=200, ny_fine=200):
    xi = np.linspace(x_vals.min(), x_vals.max(), nx_fine)
    yi = np.linspace(y_vals.min(), y_vals.max(), ny_fine)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x_vals, y_vals), values, (XI, YI), method='cubic')
    return XI, YI, ZI

XI, YI, sigma_xx_smooth = smooth_grid(x_layer, y_layer, sigma_xx_layer)
_, _, sigma_yy_smooth = smooth_grid(x_layer, y_layer, sigma_yy_layer)
_, _, sigma_xy_smooth = smooth_grid(x_layer, y_layer, sigma_xy_layer)
_, _, u_x_smooth = smooth_grid(x_layer, y_layer, u_x_layer)
_, _, u_y_smooth = smooth_grid(x_layer, y_layer, u_y_layer)
_, _, u_z_smooth = smooth_grid(x_layer, y_layer, u_z_layer)

# Mask the semi-circular hole region in the plotted grids so it appears white / transparent
hole_mask_grid = ((XI - hole_center_x)**2 + (YI - hole_center_y)**2) <= (hole_R**2)
hole_mask_grid = np.logical_and(hole_mask_grid, YI >= 0.0)  # semi-circle upwards

sigma_xx_smooth_masked = np.array(sigma_xx_smooth)
sigma_yy_smooth_masked = np.array(sigma_yy_smooth)
sigma_xy_smooth_masked = np.array(sigma_xy_smooth)
u_x_smooth_masked = np.array(u_x_smooth)
u_y_smooth_masked = np.array(u_y_smooth)
u_z_smooth_masked = np.array(u_z_smooth)

sigma_xx_smooth_masked[hole_mask_grid] = np.nan
sigma_yy_smooth_masked[hole_mask_grid] = np.nan
sigma_xy_smooth_masked[hole_mask_grid] = np.nan
u_x_smooth_masked[hole_mask_grid] = np.nan
u_y_smooth_masked[hole_mask_grid] = np.nan
u_z_smooth_masked[hole_mask_grid] = np.nan

# ----------------------------
# Colormap
# ----------------------------
colors = ["darkblue","lightblue","green","yellow","orange","red"]
cmap_rainbow = LinearSegmentedColormap.from_list("rainbow_like", colors)
cmap_rainbow.set_bad(color='white')

# ----------------------------
# Mesh lines for plotting (only valid elements)
# ----------------------------
valid_idx = np.where(valid_elems_cpu)[0]
segments = []
for ei in valid_idx:
    # for single z-layer, element bottom face nodes are first 4 nodes
    el_nodes = elements_cpu[ei]
    pts = coords_np[el_nodes[:4], :2]
    segs = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
    segments.extend(segs)

lc_mesh  = LineCollection(segments, linewidths=0.35, colors='k', alpha=0.8)

# ----------------------------
# Plot combined: KE/IE/time series + stress & displacement contours + mesh
# ----------------------------
fig, axs = plt.subplots(3,3, figsize=(20,15))

# KE/IE vs time
axs[0,0].plot(time_list, KE_list, label="KE")
axs[0,0].plot(time_list, IE_list, label="IE")
axs[0,0].set_title("Kinetic/Internal Energy vs time")
axs[0,0].set_xlabel("time [s]")
axs[0,0].set_ylabel("Energy (arb. units)")
axs[0,0].legend()

# stresses
fields_smooth = [sigma_xx_smooth_masked, sigma_yy_smooth_masked, sigma_xy_smooth_masked]
titles = ['σ_xx [MPa]','σ_yy [MPa]','σ_xy [MPa]']
for i, (field,title) in enumerate(zip(fields_smooth,titles)):
    ax = axs[0,i+1] if i<2 else axs[1,0]
    im = ax.imshow(field, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_rainbow)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

# displacements
fields_smooth = [u_x_smooth_masked, u_y_smooth_masked, u_z_smooth_masked]
titles = ['u_x [mm]','u_y [mm]','u_z [mm]']
for i, (field,title) in enumerate(zip(fields_smooth,titles)):
    ax = axs[1,i+1] if i<2 else axs[2,0]
    im = ax.imshow(field, origin='lower', extent=[0, Lx, 0, Ly], cmap=cmap_rainbow)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

# final subplot: mesh plot showing hole (white region)
ax_mesh = axs[2,2]
ax_mesh.set_title("Mesh (outline) — hole shown as missing elements")
ax_mesh.add_collection(lc_mesh)
ax_mesh.set_xlim(0, Lx); ax_mesh.set_ylim(0, Ly)
ax_mesh.set_aspect('equal', adjustable='box')

# overlay semi-circular hole outline
theta = np.linspace(0, np.pi, 200)
x_hole = hole_center_x + hole_R * np.cos(theta)
y_hole = hole_center_y + hole_R * np.sin(theta)
ax_mesh.plot(x_hole, y_hole, 'r-', linewidth=2)  # smooth circular hole outline

plt.tight_layout()
plt.show()