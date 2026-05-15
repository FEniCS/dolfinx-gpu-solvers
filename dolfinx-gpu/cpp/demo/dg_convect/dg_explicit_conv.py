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
n = 64
k = 0  # Polynomial degree should be 0 for DG with forward Euler
t_end = 1.0
num_time_steps = 1000

# Velocity field components
w_x = 1.0
w_y = 0.0

xdmf = io.XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "r")
msh = xdmf.read_mesh()
# msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)

V = fem.functionspace(msh, ("Discontinuous Lagrange", k))
W = fem.functionspace(msh, ("DG", 1, (2, )))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Create function to store solution at previous time step and interpolate
# initial condition
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]*4) * np.sin(np.pi * x[1]*4))

w = fem.Function(W)

def vel(x):
    v0 = 0.5
    r = np.sqrt(x[0]**2 + x[1]**2)
    return [-v0 * np.pi*x[1]/r * np.sin(np.pi*r/2), v0 * np.pi*x[0]/r * np.sin(np.pi*r/2)]

w.interpolate(vel)

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

# Simulation constants
delta_t = fem.Constant(msh, t_end / num_time_steps)

# Create meshtags for the inflow boundary
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

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

# Specify boundary conditions (inflow only)
inflow_bcs = {(boundary_id["Gamma_inflow"], lambda x: np.ones_like(x[0]))}

# Specify weak form of the problem
lmbda = ufl.conditional(ufl.gt(dot(w, n), 0), 1, 0)

a = inner(u / delta_t, v) * dx

# Linear form (RHS)
f = fem.Constant(msh, PETSc.ScalarType(0.0))
L = inner(f + u_n / delta_t, v) * dx  \
    - inner(2 * avg(lmbda * w * u_n), jump(v, n)) * dS \
#    - inner(lmbda * dot(w, n) * u_n, v) * ds
#    + inner(w * u_n, grad(v)) * dx


# Apply BCs
for bc in inflow_bcs:
    u_in = fem.Function(V)
    u_in.interpolate(bc[1])
    L += - inner((1 - lmbda) * dot(w, n) * u_in, v) * ds(bc[0])

a = fem.form(a)
L = fem.form(L)

A = assemble_matrix(a)
A.assemble()
b = create_vector(V)

# Create solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Interpolate into visualisation space as VTX doesn't support piecewise constants
k_vis = k if k > 0 else 1
V_vis = fem.functionspace(msh, ("Discontinuous Lagrange", k_vis))
u_n_vis = fem.Function(V_vis)
u_n_vis.interpolate(u_n)

u_file = io.VTXWriter(msh.comm, "u.bp", [u_n_vis])

# Time stepping loop
t = 0.0
u_file.write(t)
for n in range(num_time_steps):
    print(n)
    t += delta_t.value

    with b.localForm() as b_loc:
        b_loc.set(0.0)
    assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp.solve(b, u_n.x.petsc_vec)
    u_n.x.scatter_forward()

    u_n_vis.interpolate(u_n)
    u_file.write(t)

u_file.close()

# Function spaces for exact solution
V_e = fem.functionspace(msh, ("Lagrange", k + 3))

u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

# Compute errors
e_u = norm_L2(msh.comm, u_n - u_e)

if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
