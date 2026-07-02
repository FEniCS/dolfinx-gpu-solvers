import gmsh
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
gmsh.initialize()
gmsh.model.add("circle")

# Create circle (x, y, z, radius)
circle = gmsh.model.occ.addCircle(0, 0, 0, 1.0)
curve_loop = gmsh.model.occ.addCurveLoop([circle])
surface = gmsh.model.occ.addPlaneSurface([curve_loop])

gmsh.model.occ.synchronize()

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), .03)
gmsh.model.addPhysicalGroup(2, [1], 0)

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), .03)

gmsh.model.mesh.generate(2)
gmsh.write("circle.msh")

q = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

xdmf = XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "w")
xdmf.write_mesh(q.mesh)

gmsh.finalize()
