from dolfinx import mesh, fem, io
import ufl
from ufl import inner, dx, grad, dot, dS, jump, avg
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def u_e_expr(x):
    "Analytical solution to steady state pure advection problem"
    return x[0] + 1.0


def marker_inflow(x):
    "Marker for inflow boundary"
    return np.isclose(x[0], 0.0)


# Simulation parameters
k = 1           # DG1 — requires SSP-RK for stability
t_end = 2.0
num_time_steps = 400   # CFL is tighter for DG1; increase if blow-up

xdmf = io.XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "r")
msh = xdmf.read_mesh()

V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
W = fem.functionspace(msh, ("DG", 1, (2,)))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Solution at current time level; u_stage is the RK stage value used in R(·)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0] * 4) * np.sin(np.pi * x[1] * 4))
u_stage = fem.Function(V)   # updated each RK stage — lives inside the compiled form

w = fem.Function(W)


def vel(x):
    v0 = 0.5
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return [-v0 * np.pi * x[1] / r * np.sin(np.pi * r / 2),
             v0 * np.pi * x[0] / r * np.sin(np.pi * r / 2)]


w.interpolate(vel)

n_facet = ufl.FacetNormal(msh)
delta_t = fem.Constant(msh, t_end / num_time_steps)

tdim = msh.topology.dim
msh.topology.create_entities(tdim - 1)
facet_imap = msh.topology.index_map(tdim - 1)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.zeros(num_facets, dtype=np.intc)

inflow_facets = mesh.locate_entities_boundary(msh, tdim - 1, marker_inflow)
boundary_id = {"Gamma_inflow": 1}
values[inflow_facets] = boundary_id["Gamma_inflow"]
mt = mesh.meshtags(msh, tdim - 1, indices, values)

ds_meas = ufl.Measure("ds", domain=msh, subdomain_data=mt)

inflow_bcs = [(boundary_id["Gamma_inflow"], lambda x: np.ones_like(x[0]))]

lmbda = ufl.conditional(ufl.gt(dot(w, n_facet), 0), 1, 0)

# ── Mass matrix (assembled once, never changes) ──────────────────────────────
a_mass = fem.form(inner(u, v) * dx)
M = assemble_matrix(a_mass)
M.assemble()

# ── Pre-compute block-diagonal M^{-1} ────────────────────────────────────────
# For DG spaces each cell owns its DOFs exclusively, so M is block-diagonal
# with one block per cell.  We extract those blocks from the sparse PETSc
# matrix, then batch-invert them with np.linalg.inv.  Every RK stage apply
# then reduces to a single np.matmul broadcast — no sparse solver needed.

dofmap = V.dofmap
ndofs_per_cell = dofmap.dof_layout.num_dofs
num_cells_local = msh.topology.index_map(tdim).size_local

# Local DOF indices for each owned cell: shape (num_cells_local, ndofs_per_cell)
cell_dofs_local = np.array(
    [dofmap.cell_dofs(c) for c in range(num_cells_local)], dtype=np.int32
)

# PETSc getValues requires *global* row/column indices.
# For owned DOFs: global_idx = local_idx + first_owned_global_idx.
first_global = dofmap.index_map.local_range[0]
cell_dofs_global = cell_dofs_local + first_global

# Extract one (ndofs_per_cell × ndofs_per_cell) block per cell.
M_blocks = np.stack(
    [M.getValues(cell_dofs_global[c], cell_dofs_global[c])
     for c in range(num_cells_local)]
)  # shape: (num_cells_local, ndofs_per_cell, ndofs_per_cell)

# Batch-invert; for DG1 triangles these are symmetric 3×3 SPD matrices.
M_inv = np.linalg.inv(M_blocks)  # shape: (num_cells_local, ndofs_per_cell, ndofs_per_cell)


def apply_Minv(r: np.ndarray) -> np.ndarray:
    """Apply block-diagonal M^{-1} to local DOF vector r.

    DG DOFs are cell-exclusive (no inter-cell overlap for owned cells),
    so the result is assembled by direct scatter — no summation needed.
    """
    r_blocks = r[cell_dofs_local]                               # (num_cells, ndofs)
    x_blocks = np.einsum('cij,cj->ci', M_inv, r_blocks)        # batched matvec
    x = np.zeros_like(r)
    x[cell_dofs_local] = x_blocks
    return x

# ── Spatial residual R(u_stage) ───────────────────────────────────────────────
# For DG1 the volume term inner(w*u_stage, grad(v))*dx is non-zero and
# must be included.  (For DG0 grad(v)=0 cell-wise, so it vanishes.)
f = fem.Constant(msh, PETSc.ScalarType(0.0))

L_space = (
    inner(f, v) * dx
    + inner(w * u_stage, grad(v)) * dx                          # volume term (new for DG1)
    - inner(2 * avg(lmbda * w * u_stage), jump(v, n_facet)) * dS  # interior upwind flux
)

for marker_id, u_in_func in inflow_bcs:
    u_in = fem.Function(V)
    u_in.interpolate(u_in_func)
    L_space -= inner((1 - lmbda) * dot(w, n_facet) * u_in, v) * ds_meas(marker_id)

L_space_form = fem.form(L_space)
b = create_vector(V)   # reused across stages

# SSP-RK3 stage temporaries
u1 = fem.Function(V)
u2 = fem.Function(V)


def compute_Minv_R(u_input: fem.Function) -> np.ndarray:
    """Assemble R(u_input) then apply block-diagonal M^{-1}.

    Returns a numpy array of owned DOF values; caller must scatter_forward
    into a fem.Function before the next assembly.
    """
    u_stage.x.array[:] = u_input.x.array
    u_stage.x.scatter_forward()

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L_space_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # b.array is the local (owned + ghost) PETSc buffer; owned entries are
    # now complete after the reverse scatter above.
    return apply_Minv(b.array)


# ── Visualisation ─────────────────────────────────────────────────────────────
V_vis = fem.functionspace(msh, ("Discontinuous Lagrange", k))
u_n_vis = fem.Function(V_vis)
u_n_vis.interpolate(u_n)

u_file = io.VTXWriter(msh.comm, "u_rk.bp", [u_n_vis])

# ── Time-stepping: SSP-RK3 (Shu–Osher) ───────────────────────────────────────
#
#   u^(1)   = u^n + dt * M^{-1} R(u^n)
#   u^(2)   = 3/4 u^n + 1/4 [ u^(1) + dt * M^{-1} R(u^(1)) ]
#   u^{n+1} = 1/3 u^n + 2/3 [ u^(2) + dt * M^{-1} R(u^(2)) ]
#
t = 0.0
dt = delta_t.value
u_file.write(t)

for step in range(num_time_steps):
    if step % 100 == 0:
        print(f"step {step}/{num_time_steps}  t={t:.4f}")
    t += dt

    un = u_n.x.array   # view (no copy)

    # Stage 1
    k1 = compute_Minv_R(u_n)
    u1.x.array[:] = un + dt * k1
    u1.x.scatter_forward()

    # Stage 2
    k2 = compute_Minv_R(u1)
    u2.x.array[:] = 0.75 * un + 0.25 * (u1.x.array + dt * k2)
    u2.x.scatter_forward()

    # Stage 3
    k3 = compute_Minv_R(u2)
    u_n.x.array[:] = (1.0 / 3.0) * un + (2.0 / 3.0) * (u2.x.array + dt * k3)
    u_n.x.scatter_forward()

    u_n_vis.interpolate(u_n)
    u_file.write(t)

u_file.close()

# ── Error ─────────────────────────────────────────────────────────────────────
V_e = fem.functionspace(msh, ("Lagrange", k + 3))
u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

e_u = norm_L2(msh.comm, u_n - u_e)
if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
