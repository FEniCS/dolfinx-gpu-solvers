from mpi4py import MPI
import numpy as np
import ufl
import basix

from basix.ufl import element
from dolfinx import fem, mesh, la
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary

dtype = np.float64

### Python comparison for GPU linear elasticity demo

polynomial_order = 3
quadrature_degree = 2 * (polynomial_order - 1)

n = 1
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])], (n, n, n),
    CellType.tetrahedron,
    ghost_mode=GhostMode.none,
)

gdim = msh.geometry.dim
tdim = msh.topology.dim

# cell_indices = np.array([0], dtype=np.int32)
# cell_values = np.array([1], dtype=np.int32)
# cell_tags = mesh.meshtags(msh, tdim, cell_indices, cell_values)

dx = ufl.Measure(
    "dx",
    domain=msh,
    metadata={"quadrature_degree": quadrature_degree},
)

el = element(
    "Lagrange",
    msh.basix_cell(),
    polynomial_order,
    lagrange_variant=basix.LagrangeVariant.equispaced,
    shape=(gdim,),
)

V = fem.functionspace(msh, el)

dmap = V.dofmap

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

mu = 1
lmbda = 1

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

a = fem.form(ufl.inner(sigma(du), ufl.grad(v)) * dx)

# interpolated u
u = fem.Function(V)

def u_expr(x):
    values = np.zeros((gdim, x.shape[1]), dtype=dtype)
    values[0] = np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    values[1] = -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    values[2] = 0.0
    return values

np.set_printoptions(precision=3, suppress=True, linewidth=200)
u.interpolate(u_expr)

A = fem.assemble_matrix(a)

b = fem.Function(V)
b.x.array[:] = 0.0

A.mult(u.x, b.x)

print(b.x.array)
print('b=\n', la.norm(b.x))
