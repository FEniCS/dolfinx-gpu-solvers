'''
This file is used to generate the body_force.c file that is compiled 
and linked with the main.cpp file. 
It describes the finite element formula (i.e. the integral used to distribute
a continuous force among the degrees of freedom) for the load vector so that
FFCx can generate C code for DOLFINx to use later.
It gives main.cpp form_body_force_L which is a description of how to calculate 
the local load vector on one tetrahedral cell.
'''

import basix 
from basix.ufl import element
from ufl import Constant, TestFunction, FunctionSpace, Mesh, dx, inner

# create finite element used to describe mesh geometry
coordinate_element = element("Lagrange", basix.CellType.tetrahedron, 1, shape=(3,))

# create mesh using coordinate element
mesh = Mesh(coordinate_element)

# create a displacement finite element of degree 3
# has to match the c++ code i.e. same polynomial degree, same variant, same shape
element_p3 = element("Lagrange", "tetrahedron", 3, lagrange_variant=basix.LagrangeVariant.equispaced, shape=(3,))

# vector test function
V = FunctionSpace(mesh, element_p3)

# test function
v = TestFunction(V)

# force vector
B = Constant(mesh, shape=(3,))

# linear form
L = inner(B, v) * dx
forms = [L]
