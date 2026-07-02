# Copyright (C) 2025 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier: MIT
#
# UFL form file for explicit DG1 upwind advection on tets, mass matrix only.
# Compiled with ffcx to produce dg_convect.h / dg_convect.c.
#
# Spaces
# ------
#   V : DG1 vector (n components)  – solution u
#   W : DG1 vector (2 components)  – advecting velocity w

from basix.ufl import element
from ufl import (
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
    inner,
)

# ---------------------------------------------------------------------------
# Mesh and coordinate element tets
# ---------------------------------------------------------------------------
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

# ---------------------------------------------------------------------------
# Finite element spaces
# ---------------------------------------------------------------------------
n = 3
e   = element("DG", "tetrahedron", 1, shape=(n,))  # DG1
e_w = element("DG", "tetrahedron", 1, shape=(3,))  # vector DG1 (3 components)

V = FunctionSpace(mesh, e)
W = FunctionSpace(mesh, e_w)

# ---------------------------------------------------------------------------
# Trial / test functions and coefficients
# ---------------------------------------------------------------------------
u   = TrialFunction(V)
v   = TestFunction(V)

# ---------------------------------------------------------------------------
# Form m – mass-matrix diagonal
# ---------------------------------------------------------------------------
a = inner(u, v) * dx
