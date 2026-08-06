// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <fstream>

#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/la/Vector.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include "body_force.h"
#include "cg.h"
#include "jacobi.h"
#include "p_multigrid.h"
#include "util.h"
#include "load.h"

using namespace dolfinx;
namespace po = boost::program_options;

// polynomial orders from highest to lowest for p-multigrid
using Hierarchy = PMultigridHierarchy<5, 4, 2>;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

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
    std::cout << "n=" << n << "\n";
    auto part
        = mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, 2);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n},
        mesh::CellType::tetrahedron, part));

    // Create list of all cells
    std::vector<std::int32_t> cell_list_host(
        mesh->topology()->index_map(mesh->topology()->dim())->size_local());
    std::iota(cell_list_host.begin(), cell_list_host.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_host.begin(),
                                                   cell_list_host.end());

    auto boundary = [](auto x){ // x is a table of coordinates
      std::vector<std::int8_t> marker(x.extent(1), false); // create a list called marker which has one entry for each point being checked
      // mark points on the left boundary (x=0) as true
      for (std::size_t p = 0; p < x.extent(1); ++p) // loop over all points
      {
        // if (std::abs(x(0, p)) < 1e-8) // check if x coordiante is 0/close to 0
        // {
        //   marker[p] = true; // mark this point as true
        // }

        marker[p] = std::abs(x(0, p)) < 1e-8
          || std::abs(x(1, p)) < 1e-8
          || std::abs(x(2, p)) < 1e-8
          || std::abs(x(0, p) - 1.0) < 1e-8
          || std::abs(x(1, p) - 1.0) < 1e-8
          || std::abs(x(2, p) - 1.0) < 1e-8;
      }
      return marker;
    };

    Hierarchy hierarchy(mesh, cell_list, boundary);
    auto& fine_level = hierarchy.level;

    using DeviceVector = typename Hierarchy::DeviceVector;
    DeviceVector b_device(fine_level.V->dofmap()->index_map, 3);
    fine_level.assemble_body_force(b_device, detail::ConstantBodyForce<T>{T(0), T(0), T(-1.0e-3)}); // assemble body force vector with force in negative z direction

    // 0 initial guess
    DeviceVector x_device(fine_level.V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));

    DeviceVector fine_Ax(fine_level.V->dofmap()->index_map, 3);
    DeviceVector fine_residual(fine_level.V->dofmap()->index_map, 3);

    // timing 
    constexpr int runs = 1;
    constexpr int max_v_cycles = 1000;
    constexpr T residual_tolerance = T(1e-8);

    std::vector<double> times;
    times.reserve(runs);

    int v_cycles = 0;
    T initial_residual_norm = T(0);
    T final_residual_norm = T(0);

    std::cout << std::setprecision(17);

    for (int run = 0; run < runs; ++run)
    {
      // reset x to 0 for each run
      thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));
      device_synchronize();

      initial_residual_norm = residual_norm(fine_level, x_device, b_device, fine_Ax, fine_residual);
      final_residual_norm = initial_residual_norm;
      v_cycles = 0;

      const T target_residual_norm = initial_residual_norm * residual_tolerance;

      const auto start_time = std::chrono::high_resolution_clock::now();

      while (final_residual_norm > target_residual_norm && v_cycles < max_v_cycles)
      {
        hierarchy.v_cycle(x_device, b_device);
        final_residual_norm = residual_norm(fine_level, x_device, b_device, fine_Ax, fine_residual);
        ++v_cycles;
      }
      
      device_synchronize();

      const auto end_time = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double>(end_time - start_time).count());
    }

    std::cout << "Number of iterations: " << v_cycles << "\n";

    const double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average multigrid solve time: " << average_time << " seconds\n";

    std::cout << "Initial residual norm: " << initial_residual_norm << "\n";
    std::cout << "Final residual norm: " << final_residual_norm << "\n";

    auto x = std::make_shared<fem::Function<T>>(fine_level.V);

    thrust::copy(x_device.array().begin(), x_device.array().end(), x->x()->array().begin());

    std::cout << std::setprecision(17);

    std::cout << "Computed x norm = "
              << dolfinx::la::norm(*x->x()) << "\n";

    auto b = std::make_shared<fem::Function<T>>(fine_level.V);
    
    device_synchronize();

    thrust::copy(b_device.array().begin(), b_device.array().end(), b->x()->array().begin());

    std::cout << "b norm = "
              << dolfinx::la::norm(*b->x()) << "\n";

    ///// for visualisation in PARAVIEW /////
    #ifdef HAS_ADIOS2
      x->name = "displacement";
      io::VTXWriter<U> writer(
          MPI_COMM_WORLD, "cantilever.bp", {x}, "bp4");
      writer.write(0.0);
    #endif

  }

  MPI_Finalize();
  return 0;
}
