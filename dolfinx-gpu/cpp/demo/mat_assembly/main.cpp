// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

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

#include "../../include/gpu_geometry.h"

using namespace dolfinx;
using T = double;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
        mesh::CellType::tetrahedron, part));

    int degree = 2;
    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, degree,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    int qdegree = 2;
    GPUGeometry<thrust::device_vector<U>, thrust::device_vector<std::int32_t>>
        gpu_geom(mesh->geometry(), qdegree);
    int nq = gpu_geom.num_qp();

    // Prepare on-device containers for computation at quadrature points
    thrust::device_vector<U> G6_data(
        6 * nq * mesh->topology()->index_map(3)->size_local());
    thrust::device_vector<std::int32_t> cells = {0};
    gpu_geom.compute_G6(G6_data, cells);

    std::vector<U> g_cpu(G6_data.size());
    thrust::copy(G6_data.begin(), G6_data.end(), g_cpu.begin());
    std::cout << "G6=\n";
    for (int i = 0; i < 6 * nq; ++i)
      std::cout << i << ": " << g_cpu[i] << "\n";

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
