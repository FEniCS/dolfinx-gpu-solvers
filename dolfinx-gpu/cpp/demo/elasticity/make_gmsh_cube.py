import gmsh
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio

gmsh.initialize()

### create unit cube
gmsh.model.add("unstructured_cube")
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)

gmsh.model.occ.synchronize()

### add tags
gmsh.model.addPhysicalGroup(3, [box], tag=1)
gmsh.model.setPhysicalName(3, 1, "Volume")

boundary = gmsh.model.getBoundary([(3, box)], oriented=False)
boundary_tags = [entity[1] for entity in boundary]

gmsh.model.addPhysicalGroup(2, boundary_tags, tag=1)
gmsh.model.setPhysicalName(2, 1, "Boundary")

### control mesh size
n = 100
h = 1.0 / n

gmsh.option.setNumber("Mesh.MeshSizeMin", h)
gmsh.option.setNumber("Mesh.MeshSizeMax", h)

### generate mesh
gmsh.model.mesh.generate(3)

### convert to dolfinx mesh
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

mesh_name = "geometry"

if cell_tags is not None:
    cell_tags.name = "volume markers"

if facet_tags is not None:
    facet_tags.name = "facet markers"

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

### write mesh to file
with XDMFFile(MPI.COMM_WORLD, f"{mesh_name}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)

    if cell_tags is not None:
        xdmf.write_meshtags(cell_tags, mesh.geometry, geometry_xpath="/Xdmf/Domain/Grid[@Name='geometry']/Geometry")
    
    if facet_tags is not None:
        xdmf.write_meshtags(facet_tags, mesh.geometry, geometry_xpath="/Xdmf/Domain/Grid[@Name='geometry']/Geometry")

gmsh.finalize()
