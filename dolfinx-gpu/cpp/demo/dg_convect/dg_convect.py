# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: MIT
#
# UFL form file for explicit DG upwind advection on tets.
# Compiled with ffcx to produce dg_convect.h / dg_convect.c.
#
# Spaces
# ------
#   V : DG1 (piecewise-constant) scalar  – solution u
#   W : DG1 vector (2 components)        – advecting velocity w
#
# Forms
# -----
#   L : linear form – explicit Euler RHS
#         cell integral   : u_n / dt
#         interior facets : upwind flux  (dS)
#
#   m : linear form – mass-matrix diagonal
#         cell integral   : 1 / dt
#       Assembling m gives |T_i| / dt per DOF,
#       i.e. the diagonal of the DG0 mass matrix scaled by 1/dt.
#       Used to perform the trivial diagonal solve u_new = b / m.

from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    avg,
    conditional,
    dot,
    dS,
    ds,
    dx,
    FacetNormal,
    gt,
    inner,
    jump,
)

# ---------------------------------------------------------------------------
# Mesh and coordinate element tets
# ---------------------------------------------------------------------------
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

# ---------------------------------------------------------------------------
# Finite element spaces
# ---------------------------------------------------------------------------
e   = element("DG", "tetrahedron", 1)           # scalar DG1
e_w = element("DG", "tetrahedron", 1, shape=(3,))  # vector DG1 (3 components)

V = FunctionSpace(mesh, e)
W = FunctionSpace(mesh, e_w)

# ---------------------------------------------------------------------------
# Trial / test functions and coefficients
# ---------------------------------------------------------------------------
u   = TrialFunction(V)
v   = TestFunction(V)

u_n   = Coefficient(V)   # solution at the previous time step
w     = Coefficient(W)   # advecting velocity field
delta_t = Constant(mesh) # time-step size dt

# ---------------------------------------------------------------------------
# Outward facet normal and upwind selector
# ---------------------------------------------------------------------------
n = FacetNormal(mesh)

# lmbda = 1 on the outflow side of a facet, 0 on the inflow side.
# For interior facets (dS) the restriction to '+'/'-' sides is handled by
# the avg() and jump() operators below.
lmbda = conditional(gt(dot(w, n), 0), 1, 0)

# ---------------------------------------------------------------------------
# Form L – explicit Euler RHS
#
#   b_i = (u_n_i / dt) * |T_i|                                (cell)
#         - sum_F  [[ 2 avg(lambda * w * u_n) ]] . [[ v n ]]  (interior)
#
# The factor "2 avg(...)" computes the one-sided upwind flux:
#   If dot(w, n('+')) > 0 then the upwind value is u_n('+'), giving
#   lmbda('+') = 1, lmbda('-') = 0  =>  2 avg(lmbda w u_n) = w u_n('+').
# ---------------------------------------------------------------------------
L = (
    inner(u_n / delta_t, v) * dx
    - inner(2 * avg(lmbda * w * u_n), jump(v, n)) * dS
)

# ---------------------------------------------------------------------------
# Form m – mass-matrix diagonal
# ---------------------------------------------------------------------------
a = inner(u / delta_t, v) * dx
