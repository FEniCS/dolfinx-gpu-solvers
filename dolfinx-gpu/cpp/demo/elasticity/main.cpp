// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/la/Vector.h>
#include <petscsys.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/ADIOS2Writers.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "jacobi.h"
#include "p_multigrid.h"
#include "util.h"
#include "load.h"
#include "cg.h"

using namespace dolfinx;
namespace po = boost::program_options;

// polynomial orders from highest to lowest for p-multigrid
using Hierarchy = PMultigridHierarchy<2, 1>;

// p-multigrid smoothing configuration
struct SmoothingConfig{int order; int pre; int post;};

constexpr std::array smoothing_config = {
    SmoothingConfig{2, 3, 3}, // 3 pre-smoothing and 3 post-smoothing steps for P2
};

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const GPUSelection gpu_selection = select_gpu_for_rank(MPI_COMM_WORLD);
  std::cout << "MPI rank " << rank << " local rank " << gpu_selection.local_rank << " using GPU " << gpu_selection.device << "\n";

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print usage message")(
      "n", po::value<std::size_t>()->default_value(100), "Cube mesh size");

  // Parse command line options
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);

  {
    std::int32_t n = vm["n"].as<std::size_t>();
    // std::cout << "n=" << n << "\n";
    // auto part
    //     = mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, 2);
    // auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
    //     MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n},
    //     mesh::CellType::tetrahedron, part));
    
    io::XDMFFile xdmf_file(MPI_COMM_WORLD, "/home/af854/dolfinx-gpu-solvers/dolfinx-gpu/Crescendo_NX20mm.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh<U>>(xdmf_file.read_mesh(
      fem::CoordinateElement<U>(mesh::CellType::tetrahedron, 1),
      mesh::GhostMode::none,
      "mesh"));

    // Create list of all cells
    std::vector<std::int32_t> cell_list_host(
        mesh->topology()->index_map(mesh->topology()->dim())->size_local());
    std::iota(cell_list_host.begin(), cell_list_host.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_host.begin(),
                                                   cell_list_host.end());

    const int tdim = mesh->topology()->dim();
    const int fdim = tdim - 1;                                         

    auto boundary_facets = mesh::locate_entities_boundary(
      *mesh,
      fdim,
      [](auto x){ // x is a table of coordinates
      std::vector<std::int8_t> marker(x.extent(1), false); // create a list called marker which has one entry for each point being checked
            
      constexpr T xmin = T(2588.61044);
      constexpr T tol = T(1e-2);

      // mark points on the left boundary (x=0) as true
      for (std::size_t p = 0; p < x.extent(1); ++p) // loop over all points
      {
        marker[p] = std::abs(x(0, p) - xmin) < tol; // mark points on the left boundary (x=xmin) as true
      }
      return marker;
    });

    std::cout << "Number of boundary facets: "<< boundary_facets.size() << '\n';

    constexpr T E = T(1.0e9);
    constexpr T nu = T(0.3);
    constexpr T mu = E / (T(2) * (T(1) + nu));
    constexpr T lambda = E * nu / ((T(1) + nu) * (T(1) - T(2) * nu));

    solve_timings.reset();
    device_synchronize();

    const auto setup_start =std::chrono::high_resolution_clock::now();

    Hierarchy hierarchy(mesh, cell_list, boundary_facets, lambda, mu);
    
    device_synchronize();

    const auto setup_end = std::chrono::high_resolution_clock::now();

    const double hierarchy_setup_time = std::chrono::duration<double>(setup_end - setup_start).count();

    std::cout << "Hierarchy setup time: " << hierarchy_setup_time << " seconds\n";
    setup_timings.print(hierarchy_setup_time);


    using DeviceVector = typename Hierarchy::DeviceVector;

    for (const auto& config : smoothing_config)
    {
        hierarchy.set_smoothing_steps(config.order, config.pre, config.post);
    }

    auto& fine_level = hierarchy.level;

    const auto index_map = fine_level.V->dofmap()->index_map;
    const int bs = fine_level.V->dofmap()->index_map_bs();

    std::cout << "Number of P" << fine_level.order << " nodes = " << index_map->size_global() << "\n";
    std::cout << "Number of P" << fine_level.order << " DOFs = " << bs * index_map->size_global() << "\n";

    DeviceVector b_device(fine_level.V->dofmap()->index_map, 3);
    fine_level.assemble_body_force(b_device, detail::ConstantBodyForce<T>{T(0), T(0), T(-1)}); // assemble body force vector with force in negative z direction

    // 0 initial guess
    DeviceVector x_device(fine_level.V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));

    DeviceVector fine_Ax(fine_level.V->dofmap()->index_map, 3);
    DeviceVector fine_residual(fine_level.V->dofmap()->index_map, 3);

    // outer cg solver
    elasticity::CGSolver<DeviceVector> cg_solver(fine_level.V->dofmap()->index_map, 3);
    cg_solver.set_max_iterations(1000);
    cg_solver.set_tolerance(T(1e-8));

    auto pmg_preconditioner = [&hierarchy](DeviceVector& z, const DeviceVector& r)
    {
      thrust::fill(thrust::device, z.array().begin(), z.array().end(), T(0));
      hierarchy.v_cycle(z, r);
    };


    // timing 
    constexpr int runs = 1;

    std::vector<double> times;
    times.reserve(runs);

    T initial_residual_norm = T(0);
    T final_residual_norm = T(0);
    T relative_residual_norm = T(0);
    int cg_iterations = 0;

    std::cout << std::setprecision(17);

    for (int run = 0; run < runs; ++run)
    {
      // reset x to 0 for each run
      thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));
      
      device_synchronize();

      initial_residual_norm = residual_norm(fine_level, x_device, b_device, fine_Ax, fine_residual);

      const auto start_time = std::chrono::high_resolution_clock::now();

      cg_iterations = cg_solver.solve(fine_level, x_device, b_device, pmg_preconditioner);
      
      device_synchronize();

      const auto end_time = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double>(end_time - start_time).count());
    }

    final_residual_norm = residual_norm(fine_level, x_device, b_device, fine_Ax, fine_residual);
    relative_residual_norm = final_residual_norm / initial_residual_norm;

    std::cout << "Number of iterations: " << cg_iterations << "\n";

    const double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average cg + multigrid solve time: " << average_time << " seconds\n";
    solve_timings.print(average_time);
    std::cout << "Initial residual norm: " << initial_residual_norm << "\n";
    std::cout << "Final residual norm: " << final_residual_norm << "\n";
    std::cout << "Relative residual norm: " << relative_residual_norm << "\n";

    auto x2 = std::make_shared<fem::Function<T>>(fine_level.V);

    thrust::copy(x_device.array().begin(), x_device.array().end(), x2->x()->array().begin());

    std::cout << std::setprecision(17);

    std::cout << "Computed x norm = "
              << dolfinx::la::norm(*x2->x()) << "\n";

    auto b = std::make_shared<fem::Function<T>>(fine_level.V);
    
    device_synchronize();

    thrust::copy(b_device.array().begin(), b_device.array().end(), b->x()->array().begin());

    std::cout << "b norm = " << dolfinx::la::norm(*b->x()) << "\n";

    ///// for visualisation in PARAVIEW /////
    #ifdef HAS_ADIOS2
      x2->name = "displacement";
      io::VTXWriter<U> writer(
          MPI_COMM_WORLD, "engine.bp", {x2}, "bp4");
      writer.write(0.0);
    #endif

  }

  PetscFinalize();
  return 0;
}
