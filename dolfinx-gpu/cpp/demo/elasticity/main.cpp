// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//

#include <basix/finite-element.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

#include "../../include/gpu_geometry.h"
#include "elasticity.h"
#include "util.h"

#include <fstream>
#include <iomanip>

using namespace dolfinx;
namespace po = boost::program_options;
using T = float; // float or double
using U = dolfinx::scalar_value_t<T>;

constexpr int polynomial_order = 2; // 2 or 3 for P2 or P3 tetrahedra
constexpr int quadrature_degree = 2 * (polynomial_order - 1);

template <typename ContainerI>
class GPUDofMap
{
public:
  /// @brief Construct a device dofmap from a host DofMap.
  ///
  /// Copies the flattened dofmap array to the device. The shape and
  /// index map are stored but not copied to the device.
  ///
  /// @param[in] dofmap The host-side degree-of-freedom map to copy.
  GPUDofMap(const dolfinx::fem::DofMap& dofmap)
      : _dofmap(dofmap.map().data_handle(),
                dofmap.map().data_handle() + dofmap.map().size()),
        _shape({dofmap.map().extent(0), dofmap.map().extent(1)}),
        _im(dofmap.index_map)
  {
  }

  /// @brief Return the on-device dofmap data array.
  ///
  /// The array is stored in row-major order with shape
  /// `(num_cells, num_dofs_per_cell)`.
  ///
  /// @return Reference to the device container holding dof indices.
  const ContainerI& map() const { return _dofmap; }

  /// @brief Return the size of the dofmap in a given dimension.
  ///
  /// @param[in] j Dimension index (0 for number of cells, 1 for number
  /// of dofs per cell).
  /// @return Size in dimension `j`.
  std::size_t extent(int j) const { return _shape.at(j); }

  /// @brief Return the index map associated with the dofmap.
  /// @return Index map for the owned and ghost degrees-of-freedom.
  std::shared_ptr<const dolfinx::common::IndexMap> index_map() const
  {
    return _im;
  }

private:
  ContainerI _dofmap;
  std::array<std::size_t, 2> _shape;
  std::shared_ptr<const dolfinx::common::IndexMap> _im;
};

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
    int tdim = mesh->topology()->dim();
    std::vector<std::int32_t> cell_list_0(
        mesh->topology()->index_map(mesh->topology()->dim())->size_local());
    std::iota(cell_list_0.begin(), cell_list_0.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_0.begin(),
                                                  cell_list_0.end());

    // Set degree 4 to get 14 quadrature points
    GPUGeometry<thrust::device_vector<U>, thrust::device_vector<std::int32_t>>
        g_device(mesh->geometry(), quadrature_degree);
    // detJ must be computed first: compute_K9 reuses it (via wdetJ) to
    // normalize K = J^{-T} instead of re-deriving detJ itself.
    thrust::device_vector<T> wdetJ(cell_list.size()
                                   * g_device.qpoints().size());
    g_device.compute_detJ(wdetJ, cell_list);
    thrust::device_vector<T> K(cell_list.size() * 9
                               * g_device.qpoints().size());
    g_device.compute_K9(K, wdetJ, cell_list);

    // -----------------------------------------------------------------------
    // Finite element space
    // -----------------------------------------------------------------------
    auto elem_p2 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, polynomial_order,
        basix::element::lagrange_variant::equispaced,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      elem_p2, std::vector<std::size_t>{3})));
    GPUDofMap<thrust::device_vector<std::int32_t>> gpu_dofmap(*(V->dofmap()));

    const auto& dofmap = *V->dofmap();
    const auto map = dofmap.map();

    // -----------------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------------
    auto u = std::make_shared<fem::Function<T>>(V);
    auto b = std::make_shared<fem::Function<T>>(V);

    auto left_boundary = [](auto x){ // x is a table of coordinates
      std::vector<std::int8_t> marker(x.extent(1), false); // create a list called marker which has one entry for each point being checked
      // mark points on the left boundary (x=0) as true
      for (std::size_t p = 0; p < x.extent(1); ++p) // loop over all points
      {
        if (std::abs(x(0, p)) < 1e-8) // check if x coordiante is 0/close to 0
        {
          marker[p] = true; // mark this point as true
        }
      }
      return marker;
    };

    // find the clamped nodes
    std::vector<std::int32_t> bc_nodes = fem::locate_dofs_geometrical(*V, left_boundary); // in function space V, locate the dofs that are on the boundary
    // find how many nodes there are
    std::size_t num_dofs = u->x()->array().size(); // number of nodes in the mesh
    // i.e. how many scalar values are stored in u
    std::vector<std::int8_t> bc_marker_host(num_dofs, false); // create a list of bools for each node
    
    for (std::int32_t node : bc_nodes) // loop over all the clamped nodes
    {
      // mark the clamped nodes as true
      bc_marker_host[3 *node + 0] = true;
      bc_marker_host[3 *node + 1] = true;
      bc_marker_host[3 *node + 2] = true;
    }

    // copy markers to GPU
    thrust::device_vector<std::int8_t> bc_marker_device(bc_marker_host.begin(), bc_marker_host.end());
    std::cout << "Number of clamped nodes = " << bc_nodes.size() << "\n";
    std::cout << "Number of dofs = " << num_dofs << "\n";

    u->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          const std::size_t np = x.extent(1);
          std::vector<T> vals(3 * np, 0.0);
          for (std::size_t p = 0; p < np; ++p)
          {
            vals[p] = x(0,p) * std::sin(std::numbers::pi * x(0, p))
                      * std::cos(std::numbers::pi * x(1, p));
            vals[np + p] = x(0,p) * -std::cos(std::numbers::pi * x(0, p))
                           * std::sin(std::numbers::pi * x(1, p));
          }
          return {vals, {3, np}};
        });

    // Copy u to device
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));
    la::Vector<T, thrust::device_vector<T>> b_device(V->dofmap()->index_map, 3);

    // TODO: interpolate something into u

    // Tabulate basis for element
    std::size_t nq = g_device.qpoints().size() / 3;
    std::cout << "runtime nq = " << nq << "\n";
    auto shape = elem_p2.tabulate_shape(1, nq);
    std::vector<T> table(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    std::vector<T> qpoints(g_device.qpoints().size());
    thrust::copy(g_device.qpoints().begin(), g_device.qpoints().end(),
                 qpoints.begin());
    elem_p2.tabulate(1, std::span(qpoints), {nq, 3}, std::span(table));
    assert(shape.size() == 4);
    assert(shape[0] == 4);
    assert(shape[1] == nq);
    int ndofs = shape[2];
    assert(shape[3] == 1);
    thrust::device_vector<T> phi_data(table.begin(), table.end());

    int nrep = 10;
    auto start = std::chrono::high_resolution_clock::now();

    thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
    device_synchronize();

    for (int rep = 0; rep < nrep; ++rep)
    {
      assemble_elasticity_action<polynomial_order>(b_device, u_device, phi_data, K, wdetJ,
                                    gpu_dofmap.map(), cell_list, bc_marker_device);
    }

    device_synchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    double time = duration.count();
    double number_of_dofs = static_cast<double>(u_device.array().size());
    double computation_rate = nrep * number_of_dofs / (1e9 * time);
    
    std::cout << "Computation rate = " << computation_rate << " Gdofs/s\n";
    std::cout << "Number of dofs = " << number_of_dofs << "\n";

    // save result to CSV file
    const std::string filename = "elasticity_rate_vs_n.csv";

    std::ifstream check_file(filename);
    bool write_header = !check_file.good();
    check_file.close();

    std::ofstream csv(filename, std::ios::app);

    if (write_header)
    {
      csv << "n,number_of_dofs,time_s,nrep,computation_rate_Gdofs_s\n";
    }

    csv << n << "," << number_of_dofs << "," << time << "," << nrep << "," << computation_rate << "\n";

    thrust::copy(b_device.array().begin(), b_device.array().end(),
                 b->x()->array().begin());

    dolfinx::list_timings(MPI_COMM_WORLD);

    // Correctness check: run once only
    thrust::fill(b_device.array().begin(), b_device.array().end(), T(0));
    device_synchronize();

    assemble_elasticity_action<polynomial_order>(
        b_device, u_device, phi_data, K, wdetJ,
        gpu_dofmap.map(), cell_list, bc_marker_device);

    device_synchronize();

    thrust::copy(b_device.array().begin(), b_device.array().end(),
                b->x()->array().begin());

    std::cout << "b norm = " << std::setprecision(17) << dolfinx::la::norm(*b->x()) << "\n";

    // const auto& b_array = b->x()->array();

    // for (std::size_t i = 0; i < b_array.size(); ++i)
    // {
    //   std::cout << "b[" << i << "] = " << b_array[i] << "\n";
    // }
  }

  MPI_Finalize();
  return 0;
}
