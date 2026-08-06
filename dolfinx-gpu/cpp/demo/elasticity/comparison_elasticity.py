from mpi4py import MPI
import numpy as np
import ufl
import basix
import time

from basix.ufl import element
from dolfinx import fem, mesh, la, io
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary

from petsc4py import PETSc
from dolfinx.fem.petsc import (
    assemble_matrix as assemble_petsc_matrix,
    assemble_vector,
    apply_lifting
)

dtype = np.float64 # float32 or float64

### Python comparison for GPU linear elasticity demo

polynomial_order = 5 # 2 or 3 for P2 or P3 tetrahedra
quadrature_degree = 12
jacobi = True # use Jacobi preconditioner in CG

n = 1
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0], dtype=dtype), 
    np.array([1.0, 1.0, 1.0], dtype=dtype)], 
    (n, n, n),
    CellType.tetrahedron,
    ghost_mode=GhostMode.none,
    dtype=dtype
)

print("mesh dtype:", msh.geometry.x.dtype)

gdim = msh.geometry.dim
tdim = msh.topology.dim

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
    dtype=dtype
)

V = fem.functionspace(msh, el)

### adding boundary conditions
# clamp the face x = 0
u_x = u_y = u_z = 0.0

def boundary(x): # where x is an array of coordinates 
    return (np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0)
        | np.isclose(x[2], 0.0)
        | np.isclose(x[2], 1.0)
    )

facets = locate_entities_boundary(msh, tdim - 1, boundary) # finds mesh boundary entities where boundary is true
# in this case, 2D faces of the 3D cells whose coordinates satisfy x=0
bc_dofs = fem.locate_dofs_topological(V, tdim - 1, facets) # converts the boundary faces into dofs
# essentially which dofs of function space V live on these boundary facets 

zero = np.zeros(gdim, dtype=dtype) # vector to impose on boundary 
bc = fem.dirichletbc(zero, bc_dofs, V) # actually create the Dirichlet boundary condition

dmap = V.dofmap

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

mu = dtype(1)
lmbda = dtype(1)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return dtype(2.0) * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(gdim)

# exact solution for method of manufactured solutions testing
xyz = ufl.SpatialCoordinate(msh)
X = xyz[0] * (1.0 - xyz[0])
Y = xyz[1] * (1.0 - xyz[1])
Z = xyz[2] * (1.0 - xyz[2])

u_exact = ufl.as_vector((0.0, 0.0, X*Y*Z))

body_force = -ufl.div(sigma(u_exact))

a = fem.form(ufl.inner(sigma(du), ufl.grad(v)) * dx, dtype=dtype)

# # constant downwards force
# body_force = fem.Constant(msh, np.array((0.0, 0.0, -1e-3), dtype=dtype))

# load form
L = fem.form(ufl.inner(body_force, v) * dx, dtype=dtype)

# assemble elasticity matrix
A = assemble_petsc_matrix(a, bcs=[bc])
A.assemble()

# assemble force vector
b = assemble_vector(L)

# account for dirichlet boundary conditions
apply_lifting(b, [a], bcs=[[bc]])

b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# set clamped dofs to zero in the vector
bc.set(b.array)

# 0 initial guess
x = fem.Function(V, dtype=dtype)
x.x.array[:] = 0.0

# Set solver options
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.JACOBI if jacobi else PETSc.PC.Type.NONE)
solver.setTolerances(rtol=1e-8, max_it=5000)
solver.setInitialGuessNonzero(False)
solver.setUp()

runs = 1
solve_times = []

for run in range(runs):
    x.x.petsc_vec.set(0.0) # reset solution vector to zero
    start_time = time.perf_counter()
    solver.solve(b, x.x.petsc_vec)
    end_time = time.perf_counter()
    solve_times.append(end_time - start_time)

x.x.scatter_forward() # update ghost values

# comput L2 error norm
error_L2 = fem.form(ufl.inner(x - u_exact, x - u_exact) * dx, dtype=dtype)
error_squared_local = fem.assemble_scalar(error_L2)
error_squared = msh.comm.allreduce(error_squared_local, op=MPI.SUM)
l2_error = np.sqrt(error_squared)
h = 1.0 / n

print(f"Polynomial order = {polynomial_order}")
print(f"mesh size h = {h:.17e}")
print (f"L2 error = {l2_error:.17e}")

average_time = sum(solve_times) / runs
print(f"Average solve time: {average_time:.6f}")

# name shown in paraview
x.name = "displacement"

# write solution to file
writer = io.VTXWriter(msh.comm, "python_cantilever.bp", [x], "bp4")
writer.write(0.0)
writer.close()

print("Computed x norm = ", x.x.petsc_vec.norm())
print("b norm = ", b.norm())
print("number of iterations = ", solver.getIterationNumber())