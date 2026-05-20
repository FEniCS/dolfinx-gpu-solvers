// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//
// C++ implementation of an explicit DG1 upwind advection scheme on a
// tetrahedral mesh.  The variational forms are generated from dg_convect.py
// via ffcx.
//
// Time integration: SSP-RK3 (Shu–Osher three-stage scheme)
//   Stage 1:  u1       = u^n + dt * M^{-1} R(u^n)
//   Stage 2:  u1       = 0.75*u^n + 0.25*(u1 + dt * M^{-1} R(u1))
//   Stage 3:  u^{n+1}  = (u^n + 2*(u1 + dt * M^{-1} R(u1))) / 3
// where M is the block-diagonal DG1 mass matrix (4×4 blocks) and R is the
// explicit upwind convection residual assembled on interior facets.
//
// Space: DG1 (piecewise-linear) for the scalar solution u
//        DG1 vector (3 components) for the advecting velocity w

#include "block_diag.h"
#include "dg1_gpu.h"
#include "dg_convect.h"
#include "geometry.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/la/Vector.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

using namespace dolfinx;
using T = double;
using U = dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    // -----------------------------------------------------------------------
    // Simulation parameters
    // -----------------------------------------------------------------------
    constexpr int n = 50;
    constexpr double t_end = 2.0;
    constexpr int num_steps = 1000;
    constexpr double dt = t_end / num_steps;
    constexpr int io_stride = 5; // write output every this many steps

    // -----------------------------------------------------------------------
    // Mesh: box of tets, shared-facet ghost mode (needed for dS)
    // -----------------------------------------------------------------------
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto msh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.1}}}, {n, n, 5},
        mesh::CellType::tetrahedron, part));

    // Ensure facet entities and facet↔cell connectivity exist
    int tdim = msh->topology()->dim();
    msh->topology_mutable()->create_entities(tdim - 1);
    msh->topology_mutable()->create_connectivity(tdim - 1, tdim);
    msh->topology_mutable()->create_connectivity(tdim, tdim - 1);
    msh->topology_mutable()->create_entity_permutations();
    auto [normals, detJ, Kadj] = compute_facet_normals(*msh);

    int nfacets = msh->topology()->index_map(tdim - 1)->size_local();
    int ncells = msh->topology()->index_map(tdim)->size_local();
    if (ncells > 16000000)
      throw std::runtime_error("Need to use 64-bit for facet_to_cell");
    std::vector<std::int32_t> facet_to_cell_0(nfacets * 2, -1);
    std::vector<std::int32_t> facet_list_0;
    auto f_to_c = msh->topology()->connectivity(tdim - 1, tdim);
    auto c_to_f = msh->topology()->connectivity(tdim, tdim - 1);
    const std::vector<std::uint8_t>& fperm
        = msh->topology()->get_facet_permutations();
    for (int i = 0; i < f_to_c->num_nodes(); ++i)
    {
      auto cells = f_to_c->links(i);
      if (cells.size() == 2)
      {
        facet_list_0.push_back(i);
        std::int32_t c0 = cells[0] < cells[1] ? cells[0] : cells[1];
        std::int32_t c1 = cells[0] < cells[1] ? cells[1] : cells[0];
        std::span<const std::int32_t> f0 = c_to_f->links(c0);
        std::span<const std::int32_t> f1 = c_to_f->links(c1);
        for (std::int32_t j = 0; j < 4; ++j)
        {
          if (f0[j] == i)
          {
            // Combine cell-local facet and perm and place in lower 8 bits
            // packed with 6 bits for perm, 2 for facet index
            std::int32_t fnp0 = (fperm[c0 * 4 + j] << 2) | j;
            facet_to_cell_0[i * 2] = (c0 << 8) | fnp0;
          }
          if (f1[j] == i)
          {
            std::int32_t fnp1 = (fperm[c1 * 4 + j] << 2) | j;
            facet_to_cell_0[i * 2 + 1] = (c1 << 8) | fnp1;
          }
        }
      }
    }

    thrust::device_vector<std::int32_t> facet_list(facet_list_0.begin(),
                                                   facet_list_0.end());
    thrust::device_vector<std::int32_t> facet_to_cell(facet_to_cell_0.begin(),
                                                      facet_to_cell_0.end());
    std::vector<std::int32_t> cell_list_0(ncells);
    std::iota(cell_list_0.begin(), cell_list_0.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_0.begin(),
                                                  cell_list_0.end());

    // -----------------------------------------------------------------------
    // Finite element spaces
    //   V : DG1 scalar  (piecewise-linear solution)
    //   W : DG1 vector  (advecting velocity, 3 components)
    // -----------------------------------------------------------------------
    auto elem_dg0 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 0,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ true);

    auto elem_dg1 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ true);

    // Get quadrature points on a reference triangle
    auto [qpts, qwts] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::get_default_rule(basix::cell::type::triangle, 3),
        basix::cell::type::triangle, basix::polyset::type::standard, 3);

    std::cout << "qwts=";
    for (auto w : qwts)
      std::cout << w << " ";
    std::cout << std::endl;

    std::vector<T> qpoints;
    for (int j = 0; j < qpts.size() / 2; ++j)
    {
      qpoints.push_back(1 - qpts[j * 2] - qpts[j * 2 + 1]);
      qpoints.push_back(qpts[j * 2]);
      qpoints.push_back(qpts[j * 2 + 1]);
    }
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < qpts.size() / 2; ++j)
      {
        if (i % 3 == 0)
          qpoints.push_back(T(0));
        qpoints.push_back(qpts[j * 2]);
        if (i % 3 == 1)
          qpoints.push_back(T(0));
        qpoints.push_back(qpts[j * 2 + 1]);
        if (i % 3 == 2)
          qpoints.push_back(T(0));
      }
    }

    std::size_t nq = qpoints.size() / 3;
    std::cout << "nq = " << nq << "\n";

    for (int i = 0; i < nq; ++i)
    {
      std::cout << i << ": ";
      for (int j = 0; j < 3; ++j)
        std::cout << qpoints[i * 3 + j] << ", ";
      std::cout << "\n";
    }

    auto shape = elem_dg1.tabulate_shape(0, qpoints.size() / 3);
    std::vector<T> table(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    elem_dg1.tabulate(0, qpoints, {nq, 3}, std::span(table));
    // Suppress near-zeros
    std::for_each(table.begin(), table.end(),
                  [](T& q) { q = (std::abs(q) < 1e-15) ? 0.0 : q; });
    thrust::device_vector<T> phi_device(table.begin(), table.end());

    for (int i = 0; i < table.size() / 4; ++i)
    {
      std::cout << i << ": ";
      for (int j = 0; j < 4; ++j)
        std::cout << table[i * 4 + j] << ", ";
      std::cout << "\n";
    }

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            msh, std::make_shared<fem::FiniteElement<U>>(elem_dg1)));

    // Vector DG1: same scalar element, value_shape = {3} (3 spatial components)
    auto W
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            msh, std::make_shared<fem::FiniteElement<U>>(
                     elem_dg1, std::vector<std::size_t>{3})));

    // -----------------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------------
    auto u_n = std::make_shared<fem::Function<T>>(V); // previous time step
    auto w = std::make_shared<fem::Function<T>>(W);   // velocity field

    // Initial condition: sin(4 pi x) sin(4 pi y)
    u_n->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          const std::size_t np = x.extent(1);
          std::vector<T> vals(np);
          std::ranges::transform(
              std::views::iota(std::size_t(0), np), vals.begin(),
              [&x](std::size_t p)
              {
                return std::sin(4 * std::numbers::pi * x(0, p))
                       * std::sin(4 * std::numbers::pi * x(1, p));
              });
          return {vals, {np}};
        });

    // Velocity: divergence-free rotation  w = (sin(pi x)cos(pi y),
    //                                          -cos(pi x)sin(pi y))
    w->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          const std::size_t np = x.extent(1);
          std::vector<T> vals(3 * np, 0.0);
          for (std::size_t p = 0; p < np; ++p)
          {
            vals[p] = std::sin(std::numbers::pi * x(0, p))
                      * std::cos(std::numbers::pi * x(1, p));
            vals[np + p] = -std::cos(std::numbers::pi * x(0, p))
                           * std::sin(std::numbers::pi * x(1, p));
          }
          return {vals, {3, np}};
        });

    // Copy w to device
    la::Vector<T, thrust::device_vector<T>> w_device(*(w->x()));

    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    auto dt_const = std::make_shared<fem::Constant<T>>(T(dt));

    // -----------------------------------------------------------------------
    // Variational forms
    //   L_form : explicit RHS (cell + interior facets + inflow BC)
    //   m_form : mass-matrix diagonal (cell integral of c_one / dt)
    // -----------------------------------------------------------------------
    // Single-domain problems pass an empty entity_maps vector.
    fem::Form<T> L_form
        = fem::create_form<T>(*form_dg_convect_L, {V}, {{"u_n", u_n}, {"w", w}},
                              {{"delta_t", dt_const}}, {}, {});

    fem::Form<T> a_form = fem::create_form<T>(*form_dg_convect_a, {V, V}, {},
                                              {{"delta_t", dt_const}}, {}, {});

    // -----------------------------------------------------------------------
    // Assemble block mass matrix M (4×4 blocks, one per cell)
    // and immediately invert each block: returns M^{-1} in DG layout.
    // Done once; M is time-independent.
    // -----------------------------------------------------------------------
    auto map = V->dofmap()->index_map;
    const int bs = V->dofmap()->index_map_bs();
    const std::size_t nlocal = map->size_local();

    BlockDiagonalSolver block_mass(a_form, {});

    // -----------------------------------------------------------------------
    // RHS vector (re-assembled every step)
    // -----------------------------------------------------------------------
    la::Vector<T> b(map, bs);

    // Copy vectors to device
    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> un_device(*(u_n->x()));

    // -----------------------------------------------------------------------
    // Output file (ADIOS2 / VTX format, produces u.bp)
    // -----------------------------------------------------------------------
    io::VTXWriter<U> vtx(msh->comm(), "u.bp", {u_n});

    // -----------------------------------------------------------------------
    // Time-stepping loop
    // -----------------------------------------------------------------------
    double t = 0.0;
    vtx.write(t);

    bool output_gpu = true;
    for (int step = 0; step < num_steps; ++step)
    {
      std::cout << step << "\n";
      t += dt;

      auto& u_arr = u_n->x()->array();

      if (output_gpu)
      {
        // Run kernel on GPU
        // k = M⁻¹R(un)

        thrust::device_vector<T> k(un_device.array().size());
        thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
        run_dg1_convection(b_device.array(), un_device.array(),
                           w_device.array(), phi_device, normals, Kadj,
                           facet_to_cell, facet_list, cell_list);
        block_mass.solve(b_device.array(), k);

        cudaDeviceSynchronize();

        // u1 = un + k * dt
        thrust::device_vector<T> u1(un_device.array().size());
        thrust::transform(un_device.array().begin(), un_device.array().end(),
                          k.begin(), u1.begin(),
                          [dt] __device__(T x, T y) { return x + y * dt; });

        // k = M⁻¹R(u1)
        thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
        run_dg1_convection(b_device.array(), u1, w_device.array(), phi_device,
                           normals, Kadj, facet_to_cell, facet_list, cell_list);
        block_mass.solve(b_device.array(), k);

        cudaDeviceSynchronize();

        // u1 = 0.75 * un + 0.25 * (u1 + k * dt)
        thrust::transform(u1.begin(), u1.end(), k.begin(), u1.begin(),
                          [dt] __device__(T x, T y) { return x + y * dt; });
        thrust::transform(un_device.array().begin(), un_device.array().end(),
                          u1.begin(), u1.begin(), [] __device__(T x, T y)
                          { return 0.75 * x + 0.25 * y; });

        // k = M⁻¹R(u1)
        thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
        run_dg1_convection(b_device.array(), u1, w_device.array(), phi_device,
                           normals, Kadj, facet_to_cell, facet_list, cell_list);
        block_mass.solve(b_device.array(), k);

        cudaDeviceSynchronize();

        // un = (1.0/3.0) * un + (2.0/3.0) * (u1 + k * dt)
        thrust::transform(u1.begin(), u1.end(), k.begin(), u1.begin(),
                          [dt] __device__(T x, T y) { return x + y * dt; });
        thrust::transform(un_device.array().begin(), un_device.array().end(),
                          u1.begin(), un_device.array().begin(),
                          [] __device__(T x, T y)
                          { return (x + 2.0 * y) / 3.0; });
      }
      else
      {
        // Assemble RHS
        std::ranges::fill(b.array(), T(0));
        fem::assemble_vector(b.array(), L_form);
        b.scatter_rev(std::plus<T>());

        // Scatter updated local values to ghost DOFs
        u_n->x()->scatter_fwd();
      }

      if ((step + 1) % io_stride == 0)
      {
        if (output_gpu)
          thrust::copy(un_device.array().begin(), un_device.array().end(),
                       u_arr.begin());
        vtx.write(t);
      }
    }

    // Print final L2 norm as a basic sanity check
    const T local_sq = std::transform_reduce(
        u_n->x()->array().begin(),
        std::next(u_n->x()->array().begin(),
                  static_cast<std::ptrdiff_t>(nlocal)),
        T(0), std::plus<T>(), [](T v) { return v * v; });

    T global_sq = 0;

    MPI_Reduce(&local_sq, &global_sq, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, 0,
               msh->comm());

    if (dolfinx::MPI::rank(msh->comm()) == 0)
      std::cout << "||u||_l2 = " << std::sqrt(global_sq) << "\n";

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
