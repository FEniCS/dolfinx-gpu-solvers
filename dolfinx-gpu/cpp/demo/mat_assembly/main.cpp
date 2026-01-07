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
#include <thrust/inner_product.h>

#include "../../include/gpu_geometry.h"

#include "laplacian.h"
#include "sparsity.h"

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

    int degree = 1;
    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, degree,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(element)));

    // Create matrix data structure
    GPUDofMap<thrust::device_vector<std::int32_t>> gpu_dofmap(*(V->dofmap()));
    GPUSparsityPattern gpu_sparsity = create_sparsity(gpu_dofmap);
    dolfinx::la::MatrixCSR<T, thrust::device_vector<T>,
                           thrust::device_vector<std::int32_t>,
                           thrust::device_vector<std::int32_t>>
        A(gpu_sparsity);

    // Copy mesh geometry to device, and select quadrature
    int qdegree = 1;
    GPUGeometry<thrust::device_vector<U>, thrust::device_vector<std::int32_t>>
        gpu_geom(mesh->geometry(), qdegree);
    std::size_t nq = gpu_geom.qweights().size();

    // Prepare on-device containers for computation at quadrature points
    int ncells = mesh->topology()->index_map(3)->size_local();
    std::cout << "ncells = " << ncells << ", nq = " << nq << "\n";
    thrust::device_vector<U> G6_data(6 * nq * ncells);
    thrust::device_vector<std::int32_t> cells(ncells);
    thrust::copy(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(ncells), cells.begin());
    gpu_geom.compute_G6(G6_data, cells);

    // Tabulate basis for element
    auto shape = element.tabulate_shape(1, nq);
    std::vector<T> table(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    std::vector<T> qpoints(gpu_geom.qpoints().size());
    thrust::copy(gpu_geom.qpoints().begin(), gpu_geom.qpoints().end(),
                 qpoints.begin());
    element.tabulate(1, std::span(qpoints), {nq, 3}, std::span(table));
    assert(shape.size() == 4);
    assert(shape[0] == 4);
    assert(shape[1] == nq);
    int ndofs = shape[2];
    assert(shape[3] == 1);

    std::cout << "Ndofs(element) = " << ndofs << "\n";

    // Copy basis function and derivatives at qpts to phi_data
    thrust::device_vector<T> phi_data(table.begin(), table.end());

    // Assemble Laplacian on-device
    assemble_mat_laplacian(A, phi_data, G6_data, gpu_dofmap.map(), cells, nq,
                           ndofs);

    T norm = thrust::inner_product(A.values().begin(), A.values().end(),
                                   A.values().begin(), 0.0);

    std::cout << "A.norm = " << norm << "\n";

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}
