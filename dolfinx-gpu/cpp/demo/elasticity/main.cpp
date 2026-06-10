// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/la/Vector.h>

#include "elasticity.h"

using namespace dolfinx;
using T = double;
using U = dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    int n = 2;
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n},
        mesh::CellType::tetrahedron, part));

    // Create list of all cells
    int tdim = mesh->topology()->dim();
    std::vector<std::int32_t> cell_list_0(
        mesh->topology()->index_map(mesh->topology()->dim())->size_local());
    std::iota(cell_list_0.begin(), cell_list_0.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_0.begin(),
                                                  cell_list_0.end());

    // -----------------------------------------------------------------------
    // Finite element space
    // -----------------------------------------------------------------------
    auto elem_p2 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 2,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      elem_p2, std::vector<std::size_t>{3})));

    // -----------------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------------
    auto u = std::make_shared<fem::Function<T>>(V);

    // Copy u to device
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
