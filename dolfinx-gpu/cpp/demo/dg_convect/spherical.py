import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem import Function, functionspace, Expression
gmsh.initialize()
gmsh.model.add("sphere")

x0 = 0.5
x1 = 1.0

# Create circle (x, y, z, radius)
sph1 = gmsh.model.occ.addSphere(0, 0, 0, x1)
sph2 = gmsh.model.occ.addSphere(0, 0, 0, x0)

print(sph1, sph2)
cmb3 = gmsh.model.occ.cut([(3, sph1)], [(3, sph2)])

gmsh.model.occ.synchronize()

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), .02)
gmsh.model.addPhysicalGroup(3, [1], 0)

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), .02)

gmsh.model.mesh.generate(3)
gmsh.write("sphere.msh")

q = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
mesh = q.mesh

xdmf = XDMFFile(MPI.COMM_WORLD, "sphere.xdmf", "w")
xdmf.write_mesh(mesh)

DG = functionspace(mesh, ("CG", 1, (3,)))
w = Function(DG)

from ufl import SpatialCoordinate, curl, inner, sqrt
import ufl

x = SpatialCoordinate(mesh)
r = (ufl.sqrt(inner(x, x)) - x0)/(x1 - x0)
psi = ufl.as_vector((1.,1.,ufl.cos(3 *ufl.pi * x[0]))) * ufl.sin(ufl.pi * r)
w_expr = Expression(curl(psi), DG.element.interpolation_points)
w.interpolate(w_expr)

xdmf.write_function(w)

gmsh.finalize()
