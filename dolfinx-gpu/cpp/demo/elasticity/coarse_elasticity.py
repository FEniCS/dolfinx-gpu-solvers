### this file is to help assemble the sparse matrix for the AMG solve


import basix
import basix.ufl
import ufl
import numpy as np

from basix.ufl import blocked_element, wrap_element


cell = basix.CellType.tetrahedron

coord_scalar = basix.create_element(
    basix.ElementFamily.P, 
    cell, 
    1, 
    lagrange_variant=basix.LagrangeVariant.unset,
    dpc_variant=basix.DPCVariant.unset,
    discontinuous=False,
    dtype=np.float64
)

coord_element = blocked_element(wrap_element(coord_scalar), (3,))

domain = ufl.Mesh(coord_element)

E = 1.0e9
nu = 0.3

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# create a linear elasticity form for a given polynomial degree p
def elasticity_form(p):
    scalar_element = basix.create_element(
        basix.ElementFamily.P,
        cell,
        p,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        dpc_variant=basix.DPCVariant.unset,
        discontinuous=False,
        dtype=np.float64
    )

    vector_element = blocked_element(wrap_element(scalar_element), (3,))

    V = ufl.FunctionSpace(domain, vector_element)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_u = ufl.sym(ufl.grad(u))

    # lambda = mu = 1 matching GPU operator
    sigma_u = (2.0 * mu * eps_u + lam * ufl.tr(eps_u) * ufl.Identity(3))

    # this is what gets assembled into the sparse matrix
    return ufl.inner(sigma_u, ufl.grad(v)) * ufl.dx


# creates elasticity forms for polynomial degrees 1-5
a1 = elasticity_form(1)
a2 = elasticity_form(2)
a3 = elasticity_form(3)
a4 = elasticity_form(4)
a5 = elasticity_form(5)

forms = [a1, a2, a3, a4, a5]