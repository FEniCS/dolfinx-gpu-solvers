# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI
import dolfinx
from dolfinx.mesh import CellType, locate_entities_boundary
from dolfinx.fem import form, locate_dofs_topological, functionspace, assemble_matrix, assemble_vector, Function
import dolfinx.cpp as cpp
from dolfinx_gpu import gpucpp, GPUVector, GPUMatrixCSR, GPUSolver, GPUSPMV
import cupy
import numpy as np

dtype = np.float32

nx = 320
ny = 320
Bmesh = dolfinx.common.Timer(f"Create mesh {nx}x{ny}")
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(nx, ny),
    cell_type=CellType.triangle, dtype=dtype)
del(Bmesh)

V = functionspace(mesh, ("Lagrange", 1))

facets = locate_entities_boundary(
    mesh,
    dim=(mesh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`:

dofs = locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = dolfinx.fem.dirichletbc(value=dtype(0.0), dofs=dofs, V=V)

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds

a = form(a, dtype=dtype)
L = form(L, dtype=dtype)

B0 = dolfinx.common.Timer("Assemble on CPU")
A = assemble_matrix(a, [bc])
b = assemble_vector(L)
u = Function(V, dtype=dtype)
u1 = GPUVector(u.x)
del(B0)

B1 = dolfinx.common.Timer("Copy A/b to device")
A1 = GPUMatrixCSR(A)
b1 = GPUVector(b)
del(B1)

from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve

B2 = dolfinx.common.Timer("Create CSR matrix (cupy)")
Ac = csr_matrix((A1.data, A1.indices, A1.indptr))
del(B2)

spmv = GPUSPMV(A1, u1, b1)
spmv.apply()


if gpucpp.backend == 'cuda':
    B3 = dolfinx.common.Timer("Solve using cupyx.scpiy.sparse.linalg.spsolve")
    u = spsolve(Ac, b1.array)
    del(B3)
    B4 = dolfinx.common.Timer("CUDSS: Setup")
    solver = GPUSolver(A1, b1, u1)
    solver.analyze()
    solver.factorize()
    del(B4)
    B5 = dolfinx.common.Timer("CUDSS: Solve")
    solver.solve()
    del(B5)

print(u1.array)

dolfinx.common.list_timings(MPI.COMM_WORLD)
