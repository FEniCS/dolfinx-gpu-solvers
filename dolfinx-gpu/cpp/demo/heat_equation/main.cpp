// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "heat.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>

#include "../../include/gpu_cudss.h"
#include "../../include/gpu_cusparse.h"

using namespace dolfinx;
using T = double;
using U = typename dolfinx::scalar_value_t<T>;

// Then follows the definition of the coefficient functions (for $f$ and
// $g$), which are derived from the {cpp:class}`Expression` class in
// DOLFINx

// Inside the `main` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the {cpp:class}`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified
// in the form file) defined relative to this mesh, we do as follows:

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {100, 100, 100},
        mesh::CellType::tetrahedron, part));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));

    //  Next, we define the variational formulation by initializing the
    //  bilinear and linear forms ($a$, $L$) using the previously
    //  defined {cpp:class}`FunctionSpace` `V`.  Then we can create the
    //  source and boundary flux term ($f$, $g$) and attach these to the
    //  linear form.

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(0.01);
    auto u = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    fem::Form<T> a = fem::create_form<T>(*form_heat_a, {V, V}, {},
                                         {{"kappa", kappa}}, {}, {});
    fem::Form<T> L
        = fem::create_form<T>(*form_heat_L, {V}, {{"f", u}}, {}, {}, {});

    //  Now, the Dirichlet boundary condition ($u = 0$) can be created
    //  using the class {cpp:class}`DirichletBC`. A
    //  {cpp:class}`DirichletBC` takes two arguments: the value of the
    //  boundary condition, and the part of the boundary on which the
    //  condition applies. In our example, the value of the boundary
    //  condition (0) can represented using a {cpp:class}`Function`,
    //  and the Dirichlet boundary is defined by the indices of degrees
    //  of freedom to which the boundary condition applies. The
    //  definition of the Dirichlet boundary condition then looks as
    //  follows:

    // Define boundary condition

    std::vector facets = mesh::locate_entities_boundary(
        *mesh, 2,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 1.0) < eps)
              marker[p] = true;
          }
          return marker;
        });
    std::vector bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    fem::DirichletBC<T> bc(0, bdofs, V);

    u->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    //  Now, we have specified the variational forms and can consider
    //  the solution of the variational problem. First, we need to
    //  define a {cpp:class}`Function` `u` to store the solution. (Upon
    //  initialization, it is simply set to the zero function.) Next, we
    //  can call the `solve` function with the arguments `a == L`, `u`
    //  and `bc` as follows:

    la::SparsityPattern sp = fem::create_sparsity_pattern(a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    dolfinx::common::Timer ta1("CPU: assemble A");
    fem::assemble_matrix(A.mat_add_values(), a, {bc});
    A.scatter_rev();
    fem::set_diagonal<T>(A.mat_set_values(), *V, {bc});
    ta1.stop();
    ta1.flush();

    // Copy matrix to device
    la::MatrixCSR<T, thrust::device_vector<T>,
                  thrust::device_vector<std::int32_t>,
                  thrust::device_vector<std::int32_t>>
        A_device(A);

    // Re-assemble as mass matrix
    dolfinx::common::Timer ta2("CPU: assemble mass");
    kappa->value[0] = 0.0;
    la::MatrixCSR<T> Amass(sp);
    fem::assemble_matrix(Amass.mat_add_values(), a, {bc});
    Amass.scatter_rev();
    fem::set_diagonal<T>(Amass.mat_set_values(), *V, {bc});
    ta2.stop();
    ta2.flush();

    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    bc.set(b.array(), std::nullopt);
    std::cout << "u.norm = " << dolfinx::la::norm(*(u->x())) << "\n";

    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));

    // Copy mass matrix to device and set up with cusparse
    la::MatrixCSR<T, thrust::device_vector<T>,
                  thrust::device_vector<std::int32_t>,
                  thrust::device_vector<std::int32_t>>
        Amass_device(Amass);
    dolfinx::la::cuda::cusparseMatVec spmv(Amass_device, b_device, u_device);

    dolfinx::la::cuda::cudssSolver solver(A_device, b_device, u_device);

    dolfinx::common::Timer tsolve1("Solve CUDSS - analysis");
    solver.analyze();
    tsolve1.stop();
    tsolve1.flush();

    dolfinx::common::Timer tsolve2("Solve CUDSS - factorisation");
    solver.factorize();
    tsolve2.stop();
    tsolve2.flush();

    //---------------------------------------------------------------------------------
    io::XDMFFile file(MPI_COMM_WORLD, "u.xdmf", "w");
    file.write_mesh(*mesh);

    // Solving the system
    for (int i = 0; i < 100; ++i)
    {
      dolfinx::common::Timer tsolve3("Solve A.u=b");
      solver.solve();
      tsolve3.stop();
      tsolve3.flush();

      dolfinx::common::Timer tsolve5("Apply spmv");
      spmv.apply();
      tsolve5.stop();
      tsolve5.flush();

      if (i % 10 == 0)
      {
        dolfinx::common::Timer tsolve4("Copy back and I/O");
        // Copy solution back to CPU
        thrust::copy(u_device.array().begin(), u_device.array().end(),
                     u->x()->array().begin());

        std::cout << "u.norm = " << dolfinx::la::norm(*(u->x())) << "\n";
        file.write_function<T>(*u, static_cast<double>(i));
      }
    }

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
