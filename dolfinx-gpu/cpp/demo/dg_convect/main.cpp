// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//
// C++ implementation of an explicit DG0 upwind advection scheme on the unit
// square.  The variational forms are generated from dg_convect.py via ffcx.
//
// Time integration: forward Euler
//   M u^{n+1} = b(u^n)
// where M is the diagonal DG0 mass matrix (scaled by 1/dt) and b is the
// explicit RHS assembled from form L.  Because M is strictly diagonal for
// DG0 the solve reduces to element-wise division.
//
// Space: DG0 (piecewise-constant) for the scalar solution u
//        DG0 vector (2 components) for the advecting velocity w

#include "dg0_gpu.h"
#include "dg_convect.h"
#include "geometry.h"
#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
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
    constexpr int n = 80;
    constexpr double t_end = 2.0;
    constexpr int num_steps = 1000;
    constexpr double dt = t_end / num_steps;
    constexpr int io_stride = 5; // write output every this many steps

    // -----------------------------------------------------------------------
    // Mesh: box of tets, shared-facet ghost mode (needed for dS)
    // -----------------------------------------------------------------------
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto msh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.1}}}, {n, n, n / 10},
        mesh::CellType::tetrahedron, part));

    // Ensure facet entities and facet↔cell connectivity exist
    int tdim = msh->topology()->dim();
    msh->topology_mutable()->create_entities(tdim - 1);
    msh->topology_mutable()->create_connectivity(tdim - 1, tdim);
    msh->topology_mutable()->create_connectivity(tdim, tdim - 1);

    // Tabulate basis function derivatives at facet midpoints
    auto element = msh->geometry().cmap();
    std::size_t nq = 4;
    std::vector<T> qpoints = {0,     1 / 3, 1 / 3, 1 / 3, 0,     1 / 3,
                              1 / 3, 1 / 3, 0,     1 / 3, 1 / 3, 1 / 3};
    auto shape = element.tabulate_shape(1, nq);
    std::vector<T> table(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    element.tabulate(1, qpoints, {nq, 3}, std::span(table));
    thrust::device_vector<T> phi_device(table.begin(), table.end());

    // Prepare facet data on CPU
    std::span<const T> phi(table.begin(), table.size());
    auto [normals, detJ] = compute_facet_normals(*msh, phi);
    int nfacets = msh->topology()->index_map(tdim - 1)->size_local();
    std::vector<std::int32_t> facet_to_cell_0(nfacets * 2, -1);
    std::vector<std::int32_t> facet_list_0;
    auto f_to_c = msh->topology()->connectivity(tdim - 1, tdim);
    for (int i = 0; i < f_to_c->num_nodes(); ++i)
    {
      auto cells = f_to_c->links(i);
      if (cells.size() == 2)
      {
        facet_list_0.push_back(i);
        facet_to_cell_0[i * 2] = cells[0] < cells[1] ? cells[0] : cells[1];
        facet_to_cell_0[i * 2 + 1] = cells[0] < cells[1] ? cells[1] : cells[0];
      }
    }
    thrust::device_vector<std::int32_t> facet_list(facet_list_0.begin(),
                                                   facet_list_0.end());
    thrust::device_vector<std::int32_t> facet_to_cell(facet_to_cell_0.begin(),
                                                      facet_to_cell_0.end());
    std::vector<std::int32_t> cell_list_0(
        msh->topology()->index_map(msh->topology()->dim())->size_local());
    std::iota(cell_list_0.begin(), cell_list_0.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_0.begin(),
                                                  cell_list_0.end());

    // -----------------------------------------------------------------------
    // Finite element spaces
    //   V : DG0 scalar  (piecewise-constant solution)
    //   W : DG0 vector  (advecting velocity, 2 components)
    // -----------------------------------------------------------------------
    auto elem_dg0 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 0,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ true);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            msh, std::make_shared<fem::FiniteElement<U>>(elem_dg0)));

    // Vector DG0: same scalar element, value_shape = {2}
    auto W
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            msh, std::make_shared<fem::FiniteElement<U>>(
                     elem_dg0, std::vector<std::size_t>{3})));

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
    auto w0_const = std::make_shared<fem::Constant<T>>(T(1));

    // -----------------------------------------------------------------------
    // Variational forms
    //   L_form : explicit RHS (cell + interior facets + inflow BC)
    //   m_form : mass-matrix diagonal (cell integral of c_one / dt)
    // -----------------------------------------------------------------------
    // Single-domain problems pass an empty entity_maps vector.
    fem::Form<T> L_form = fem::create_form<T>(
        *form_dg_convect_L, {V}, {{"u_n", u_n}, {"w", w}},
        {{"delta_t", dt_const}, {"w0", w0_const}}, {}, {});

    fem::Form<T> m_form = fem::create_form<T>(*form_dg_convect_m, {V}, {},
                                              {{"delta_t", dt_const}}, {}, {});

    // -----------------------------------------------------------------------
    // Assemble mass-matrix diagonal  M[i] = |T_i| / dt
    // (done once; M is time-independent)
    // -----------------------------------------------------------------------
    auto map = V->dofmap()->index_map;
    const int bs = V->dofmap()->index_map_bs();
    const std::size_t nlocal = map->size_local();

    la::Vector<T> M(map, bs);
    std::ranges::fill(M.array(), T(0));
    fem::assemble_vector(M.array(), m_form);
    M.scatter_rev(std::plus<T>());

    // Inverse diagonal for the explicit solve  u_new = b / M
    // array() returns std::vector<T>&; wrap in std::span to use .first().
    std::vector<T> inv_M(nlocal);
    std::ranges::transform(std::span(M.array()).first(nlocal), inv_M.begin(),
                           [](T v) { return T(1) / v; });

    // -----------------------------------------------------------------------
    // RHS vector (re-assembled every step)
    // -----------------------------------------------------------------------
    la::Vector<T> b(map, bs);

    // Copy vectors to device
    la::Vector<T, thrust::device_vector<T>> b_device(b);
    la::Vector<T, thrust::device_vector<T>> un_device(*(u_n->x()));

    // -----------------------------------------------------------------------
    // Output file
    // -----------------------------------------------------------------------
    io::XDMFFile xdmf(msh->comm(), "u.xdmf", "w");
    xdmf.write_mesh(*msh);

    // -----------------------------------------------------------------------
    // Time-stepping loop
    // -----------------------------------------------------------------------
    double t = 0.0;
    xdmf.write_function(*u_n, t);

    bool output_gpu = true;
    for (int step = 0; step < num_steps; ++step)
    {
      std::cout << step << "\n";
      w0_const->value[0] = sin(static_cast<double>(step)
                               / static_cast<double>(num_steps) * M_PI * 4);

      t += dt;

      // Run kernel on GPU
      thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
      run_dg0_convection(b_device.array(), un_device.array(), w_device.array(),
                         phi_device, normals, detJ, facet_to_cell, facet_list,
                         cell_list, dt);

      // // Assemble RHS
      // std::ranges::fill(b.array(), T(0));
      // fem::assemble_vector(b.array(), L_form);
      // b.scatter_rev(std::plus<T>());

      // // Diagonal solve: u_new[i] = b[i] / M[i]
      auto& u_arr = u_n->x()->array();
      // std::ranges::transform(std::views::iota(std::size_t(0), nlocal),
      //                        u_arr.begin(), [&](std::size_t i)
      //                        { return inv_M[i] * b.array()[i]; });

      // // Scatter updated local values to ghost DOFs
      // u_n->x()->scatter_fwd();

      if ((step + 1) % io_stride == 0)
      {
        if (output_gpu)
          thrust::copy(un_device.array().begin(), un_device.array().end(),
                       u_arr.begin());
        xdmf.write_function(*u_n, t);
      }
    }

    xdmf.close();

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
