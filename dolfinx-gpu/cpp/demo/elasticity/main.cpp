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
#include <basix/interpolation.h>
#include <dolfinx/io/ADIOS2Writers.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "../../include/gpu_geometry.h"
#include "elasticity.h"
#include "cg.h"
#include "p_transfer.h"
#include "util.h"

#include <iomanip>
#include <cmath>

using namespace dolfinx;
namespace po = boost::program_options;
using T = double; // float or double
using U = dolfinx::scalar_value_t<T>;

constexpr int polynomial_order = 3; // 2 or 3 for P2 or P3 tetrahedra
constexpr int quadrature_degree = detail::elasticity_traits<polynomial_order>::quadrature_degree;
constexpr bool jacobi = true; // use Jacobi preconditioner in CG

static_assert(polynomial_order >= 2, "Transfer requires a lower-order level");
constexpr int coarse_order = polynomial_order - 1;

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

template <typename Scalar>
struct invert_jacobi_diagonal{
  __host__ __device__ Scalar operator()(Scalar diagonal, std::int8_t bc_marker) const{
    if (bc_marker)
      return Scalar(1); // set the value to 1 if it is clamped
    else
      return Scalar(1) / diagonal; // invert the diagonal value
  }
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
    std::vector<std::int32_t> cell_list_0(
        mesh->topology()->index_map(mesh->topology()->dim())->size_local());
    std::iota(cell_list_0.begin(), cell_list_0.end(), 0);
    thrust::device_vector<std::int32_t> cell_list(cell_list_0.begin(),
                                                  cell_list_0.end());

    // contruct geometry data using quadrature degree
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
    
    // fine function space
    auto elem = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, polynomial_order,
        basix::element::lagrange_variant::equispaced,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      elem, std::vector<std::size_t>{3})));
    GPUDofMap<thrust::device_vector<std::int32_t>> gpu_dofmap(*(V->dofmap()));

    // coarse function space
    auto elem_coarse = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::tetrahedron, coarse_order,
      basix::element::lagrange_variant::equispaced,
      basix::element::dpc_variant::unset,
      /* discontinuous = */ false
    );

    auto V_coarse = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace<U>(mesh, std::make_shared<fem::FiniteElement<U>>(
        elem_coarse, std::vector<std::size_t>{3}))
    );

    // GPU dofmap for coarse function space
    GPUDofMap<thrust::device_vector<std::int32_t>> gpu_dofmap_coarse(*(V_coarse->dofmap()));

    // build local coarse to fine interpolation matrix
    auto [P_host, P_shape] = basix::compute_interpolation_operator(elem_coarse, elem);

    // check the interpolation matrix has the expected shape
    constexpr int coarse_dofs_kernel = detail::elasticity_traits<coarse_order>::ndofs;
    constexpr int fine_dofs_kernel = detail::elasticity_traits<polynomial_order>::ndofs;

    if (P_shape[0] != fine_dofs_kernel || P_shape[1] != coarse_dofs_kernel){
      throw std::runtime_error("Unexpected interpolation matrix dimensions");
    }

    std::cout << "Interpolation matrix shape = "
              << P_shape[0] << " x "
              << P_shape[1] << "\n";

    // copy interpolation matrix to GPU
    thrust::device_vector<T> P_device(P_host.begin(), P_host.end());

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
    std::size_t num_dofs = u->x()->array().size(); // number of scalar entries stored in u
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
    using DeviceVector = decltype(b_device);

    // test functions for interpolation from coarse to fine function space
    auto test_field = [](auto x)
        -> std::pair<std::vector<T>, std::vector<std::size_t>>
    {
      const std::size_t np = x.extent(1);
      std::vector<T> vals(3 * np, T(0));

      for (std::size_t p = 0; p < np; ++p)
      {
        const T X = x(0, p);
        const T Y = x(1, p);
        const T Z = x(2, p);

        if constexpr (polynomial_order == 2)
        {
          // P1 to P2 with a linear field
          vals[p]          = X;
          vals[np + p]     = Y;
          vals[2 * np + p] = Z;
        }
        else if constexpr (polynomial_order == 3)
        {
          // P2 to P3 test with a quadratic field
          vals[p]          = X * X + Y * Z;
          vals[np + p]     = Y * Y + X * Z;
          vals[2 * np + p] = Z * Z + X * Y;
        }
      }

      return {vals, {3, np}};
    };    
    

    auto u_coarse = std::make_shared<fem::Function<T>>(V_coarse);
    auto u_fine = std::make_shared<fem::Function<T>>(V);
    u_coarse->interpolate(test_field);
    u_fine->interpolate(test_field);

    la::Vector<T, thrust::device_vector<T>> coarse_device(*(u_coarse->x()));
    DeviceVector fine_from_coarse(V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, fine_from_coarse.array().begin(), fine_from_coarse.array().end(), T(0));


    // Tabulate basis for element
    std::size_t nq = g_device.qpoints().size() / 3;
    auto shape = elem.tabulate_shape(1, nq);
    std::vector<T> table(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    std::vector<T> qpoints(g_device.qpoints().size());
    thrust::copy(g_device.qpoints().begin(), g_device.qpoints().end(),
                 qpoints.begin());
    elem.tabulate(1, std::span(qpoints), {nq, 3}, std::span(table));
    assert(shape.size() == 4);
    assert(shape[0] == 4);
    assert(shape[1] == nq);
    int ndofs = shape[2];
    assert(shape[3] == 1);
    thrust::device_vector<T> phi_data(table.begin(), table.end());

    auto A = [&](DeviceVector& output, const DeviceVector& input){
      thrust::fill(thrust::device, output.array().begin(), output.array().end(), T(0));

      assemble_elasticity_action<polynomial_order>(output, input, phi_data, K, wdetJ,
        gpu_dofmap.map(), cell_list, bc_marker_device);
    };

    // construct b = A*u_exact
    A(b_device, u_device);

    // 0 initial guess
    DeviceVector x_device(V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));

    // create CG solver
    elasticity::CGSolver<DeviceVector> cg_solver(V->dofmap()->index_map, 3);
    cg_solver.set_max_iterations(5000);
    cg_solver.set_tolerance(T(1e-8));

    if (jacobi){
      DeviceVector diagonal_inverse(V->dofmap()->index_map, 3);
      thrust::fill(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(), T(0));
      
      assemble_elasticity_diagonal<polynomial_order>(diagonal_inverse, phi_data, K, wdetJ,
        gpu_dofmap.map(), cell_list, bc_marker_device);
      
      // convert diag(A) to diag(A)^{-1}
      thrust::transform(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(),
        bc_marker_device.begin(), diagonal_inverse.array().begin(), invert_jacobi_diagonal<T>());

      // copy diag(A)^{-1} to CG solver
      cg_solver.set_diag_inverse(diagonal_inverse);
    }

    const int num_iterations = cg_solver.solve(A, x_device, b_device, jacobi);

    device_synchronize();

    auto x = std::make_shared<fem::Function<T>>(V);

    thrust::copy(x_device.array().begin(), x_device.array().end(), x->x()->array().begin());
    thrust::copy(b_device.array().begin(), b_device.array().end(), b->x()->array().begin());

    std::cout << std::setprecision(17);

    std::cout << "Exact u norm = "
              << dolfinx::la::norm(*u->x()) << "\n";

    std::cout << "Computed x norm = "
              << dolfinx::la::norm(*x->x()) << "\n";

    std::cout << "b norm = "
              << dolfinx::la::norm(*b->x()) << "\n";

    std::cout << "Number of iterations = "
              << num_iterations << "\n";

    // testing
    const int test_cell = 0;

    dim3 transfer_block(fine_dofs_kernel, 3);

    p_transfer::prolong_one_cell<T, coarse_dofs_kernel, fine_dofs_kernel><<<1, transfer_block>>>(
      test_cell,
      P_device.data().get(),
      gpu_dofmap_coarse.map().data().get(),
      gpu_dofmap.map().data().get(),
      coarse_device.array().data().get(),
      fine_from_coarse.array().data().get()
    );

    device_synchronize();

    // copy the fine dofmap to host
    const auto fine_dofmap_host = V->dofmap()->map();
    // get the exact values of the fine function
    const auto fine_exact_values = u_fine->x()->array();
    
    std::vector<T> fine_from_coarse_host(fine_from_coarse.array().size());
    thrust::copy(fine_from_coarse.array().begin(), fine_from_coarse.array().end(), fine_from_coarse_host.begin());

    constexpr T tolerance = T(1e-12);
    T max_error = T(0);
    int failed_values = 0;

    // loop over fine dofs and compare the computed values from the coarse function to the exact values from the fine function
    for (int fine_i = 0; fine_i < fine_dofs_kernel; ++fine_i)
    {
      const std::int32_t fine_node = fine_dofmap_host(test_cell, fine_i);

      // loop over the 3 components of the vector field
      for (int component = 0; component < 3; ++component)
      {
        const std::int32_t dof = 3 * fine_node + component;

        const T computed = fine_from_coarse_host[dof];
        const T exact = fine_exact_values[dof];
        const T error = std::abs(computed - exact);

        max_error = std::max(max_error, error);

        if (error > tolerance){

          failed_values++;
          
          std::cout << "local dof " << fine_i
                    << ", component " << component
                    << ", error = " << std::scientific
                    << std::setprecision(3) << error << '\n';
        }
      }
    }

    ///// for visualisation in PARAVIEW /////
    // dolfinx function to hold the prolonged values for visualisation
    auto u_prolonged = std::make_shared<fem::Function<T>>(V);
    // copy GPU result into dolfinx function
    std::copy(fine_from_coarse_host.begin(), fine_from_coarse_host.end(), u_prolonged->x()->array().begin());
    
    u_fine->name = "direct_fine";
    u_prolonged->name = "prolonged_fine";
    u_fine->x()->scatter_fwd();
    u_prolonged->x()->scatter_fwd();
    
    #ifdef HAS_ADIOS2
    io::VTXWriter<U> transfer_writer(MPI_COMM_WORLD, "p_transfer.bp", {u_fine, u_prolonged}, "bp4");
    transfer_writer.write(0.0);
    #endif

    std::cout << "Maximum prolongation error = " << max_error << "\n";
    std::cout << "Number of failed values = " << failed_values << "\n";
  }

  MPI_Finalize();
  return 0;
}
