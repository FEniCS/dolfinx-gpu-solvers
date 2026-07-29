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
#include <dolfinx/fem/Constant.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "../../include/gpu_geometry.h"
#include "elasticity.h"
#include "cg.h"
#include "p_transfer.h"
#include "util.h"
#include "body_force.h"

#include <iomanip>
#include <cmath>
#include <ranges>

using namespace dolfinx;
namespace po = boost::program_options;

using T = double; // float or double
using U = dolfinx::scalar_value_t<T>;

constexpr int polynomial_order = 3; // 2 or 3 for P2 or P3 tetrahedra
constexpr int quadrature_degree = detail::elasticity_traits<polynomial_order>::quadrature_degree;
constexpr int coarse_order = polynomial_order - 1;
constexpr int coarse_quadrature_degree = detail::elasticity_traits<coarse_order>::quadrature_degree;
constexpr bool jacobi_cg = true; // use Jacobi preconditioner in CG


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


// Jacobi smoother
// one weighted jacobi step: x = x + omega * D^{-1} * (b - Ax)
template <typename Vector, typename Operator>
void jacobi_smooth(
  Operator& A,
  Vector& x,
  const Vector& b,
  const Vector& diagonal_inverse,
  Vector& Ax,
  Vector& residual,
  int num_steps,
  typename Vector::value_type omega)
{
  using Scalar = typename Vector::value_type;

  for (int step = 0; step < num_steps; ++step){
    A(Ax, x); // compute Ax

    // compute residual = b - Ax
    thrust::transform(
      thrust::device,
      b.array().begin(),
      b.array().end(),
      Ax.array().begin(),
      residual.array().begin(),
      thrust::minus<Scalar>()
    );

    // multiply residual by inverse diagonal
    thrust::transform(
      thrust::device,
      residual.array().begin(),
      residual.array().end(),
      diagonal_inverse.array().begin(),
      residual.array().begin(),
      thrust::multiplies<Scalar>()
    );

    // add result to x with damping factor omega
    thrust::transform(
      thrust::device,
      residual.array().begin(),
      residual.array().end(),
      x.array().begin(),
      x.array().begin(),
      elasticity::axpyOperation<Scalar>{omega}
    );
  }
};


// residual norm helper function
template <typename Vector, typename Operator>
typename Vector::value_type residual_norm(
  Operator& A,
  const Vector& x,
  const Vector& b,
  Vector& Ax,
  Vector& residual)
{
  using Scalar = typename Vector::value_type;

  A(Ax, x); // compute Ax

  // compute residual = b - Ax
  thrust::transform(
    thrust::device,
    b.array().begin(),
    b.array().end(),
    Ax.array().begin(),
    residual.array().begin(),
    thrust::minus<Scalar>()
  );

  // compute norm of residual
  const Scalar norm_squared = thrust::inner_product(
    thrust::device,
    residual.array().begin(),
    residual.array().end(),
    residual.array().begin(),
    Scalar(0)
  );

  return std::sqrt(norm_squared);
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


    // FINE geometry data
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


    // COARSE geometry data with lower order quadrature
    // contruct geometry data using quadrature degree
    GPUGeometry<thrust::device_vector<U>, thrust::device_vector<std::int32_t>>
        g_device_coarse(mesh->geometry(), coarse_quadrature_degree);
    // detJ must be computed first: compute_K9 reuses it (via wdetJ) to
    // normalize K = J^{-T} instead of re-deriving detJ itself.
    thrust::device_vector<T> wdetJ_coarse(cell_list.size()
                                   * g_device_coarse.qpoints().size());
    g_device_coarse.compute_detJ(wdetJ_coarse, cell_list);
    thrust::device_vector<T> K_coarse(cell_list.size() * 9
                               * g_device_coarse.qpoints().size());
    g_device_coarse.compute_K9(K_coarse, wdetJ_coarse, cell_list);


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

    // std::cout << "Interpolation matrix shape = "
    //           << P_shape[0] << " x "
    //           << P_shape[1] << "\n";

    // copy interpolation matrix to GPU
    thrust::device_vector<T> P_device(P_host.begin(), P_host.end());

    // -----------------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------------
    // auto u = std::make_shared<fem::Function<T>>(V);
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


    // FINE boundary condition marker
    // find the clamped nodes
    std::vector<std::int32_t> bc_nodes = fem::locate_dofs_geometrical(*V, left_boundary); // in function space V, locate the dofs that are on the boundary
    // find how many nodes there are
    std::size_t num_dofs = b->x()->array().size(); // number of scalar entries stored in u
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
    // std::cout << "Number of clamped nodes = " << bc_nodes.size() << "\n";
    // std::cout << "Number of dofs = " << num_dofs << "\n";


    // COARSE boundary condition marker
    // find the clamped nodes
    std::vector<std::int32_t> bc_nodes_coarse = fem::locate_dofs_geometrical(*V_coarse, left_boundary); // in function space V_coarse, locate the dofs that are on the boundary
    // find how many nodes there are
    auto coarse_size_function = std::make_shared<fem::Function<T>>(V_coarse);
    std::size_t num_dofs_coarse = coarse_size_function->x()->array().size(); // number of scalar entries stored in u
    std::vector<std::int8_t> bc_marker_coarse_host(num_dofs_coarse, false); // create a list of bools for each node
    
    for (std::int32_t node : bc_nodes_coarse) // loop over all the clamped nodes
    {
      // mark the clamped nodes as true
      bc_marker_coarse_host[3 *node + 0] = true;
      bc_marker_coarse_host[3 *node + 1] = true;
      bc_marker_coarse_host[3 *node + 2] = true;
    }

    // copy markers to GPU
    thrust::device_vector<std::int8_t> bc_marker_coarse_device(bc_marker_coarse_host.begin(), bc_marker_coarse_host.end());    


    // const force [0 0 -0.001]
    auto body_force = std::make_shared<fem::Constant<T>>(std::array<T, 3>{T(0), T(0), T(-1e-3)});

    // attach runtime function space and force to the form generated by the body_force.py file
    fem::Form<T> load_form = fem::create_form<T>(
      *form_body_force_L, 
      {V}, // test function space
      {}, // no coefficient function spaces
      {{"B", body_force}}, // UFL constant named B
      {}, {}
    );

    // start with 0 host vector for b
    std::ranges:: fill(b->x()->array(), T(0));
    // assemble b_i = integral(B * phi_i)dx for all i
    fem::assemble_vector(b->x()->array(), load_form);

    // set the clamped nodes to 0 in b
    for (std::int32_t node : bc_nodes)
    {
      b->x()->array()[3 * node + 0] = T(0);
      b->x()->array()[3 * node + 1] = T(0);
      b->x()->array()[3 * node + 2] = T(0);
    }

    // u->interpolate(
    //     [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    //     {
    //       const std::size_t np = x.extent(1);
    //       std::vector<T> vals(3 * np, 0.0);
    //       for (std::size_t p = 0; p < np; ++p)
    //       {
    //         vals[p] = x(0,p) * std::sin(std::numbers::pi * x(0, p))
    //                   * std::cos(std::numbers::pi * x(1, p));
    //         vals[np + p] = x(0,p) * -std::cos(std::numbers::pi * x(0, p))
    //                        * std::sin(std::numbers::pi * x(1, p));
    //       }
    //       return {vals, {3, np}};
    //     });

    // Copy u to device
    // la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));
    la::Vector<T, thrust::device_vector<T>> b_device(*(b->x()));
    using DeviceVector = decltype(b_device);


    // FINE 
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


    // COARSE
    // Tabulate basis for element
    std::size_t nq_coarse = g_device_coarse.qpoints().size() / 3;
    auto shape_coarse = elem.tabulate_shape(1, nq_coarse);
    std::vector<T> table_coarse(
        std::accumulate(shape_coarse.begin(), shape_coarse.end(), 1, std::multiplies<int>()));
    std::vector<T> qpoints_coarse(g_device_coarse.qpoints().size());
    thrust::copy(g_device_coarse.qpoints().begin(), g_device_coarse.qpoints().end(),
                 qpoints_coarse.begin());
    elem.tabulate(1, std::span(qpoints_coarse), {nq_coarse, 3}, std::span(table_coarse));
    assert(shape_coarse.size() == 4);
    assert(shape_coarse[0] == 4);
    assert(shape_coarse[1] == nq_coarse);
    int ndofs_coarse = shape_coarse[2];
    assert(shape_coarse[3] == 1);
    thrust::device_vector<T> phi_data_coarse(table_coarse.begin(), table_coarse.end());


    // FINE elasticity operator
    auto A = [&](DeviceVector& output, const DeviceVector& input){
      thrust::fill(thrust::device, output.array().begin(), output.array().end(), T(0));

      assemble_elasticity_action<polynomial_order>(output, input, phi_data, K, wdetJ,
        gpu_dofmap.map(), cell_list, bc_marker_device);
    };


    // COARSE elasticity operator
    auto A_coarse = [&](DeviceVector& output, const DeviceVector& input){
      thrust::fill(thrust::device, output.array().begin(), output.array().end(), T(0));

      assemble_elasticity_action<coarse_order>(output, input, phi_data_coarse, K_coarse, wdetJ_coarse,
        gpu_dofmap_coarse.map(), cell_list, bc_marker_coarse_device);
    };

    // testing coarse operator
    DeviceVector coarse_input(V_coarse->dofmap()->index_map, 3);
    thrust::fill(thrust::device, coarse_input.array().begin(), coarse_input.array().end(), T(0));
    DeviceVector coarse_output(V_coarse->dofmap()->index_map, 3);

    A_coarse(coarse_output, coarse_input);
    device_synchronize();

    const T coarse_outout_squared = thrust::inner_product(
      thrust::device,
      coarse_output.array().begin(),
      coarse_output.array().end(),
      coarse_output.array().begin(),
      T(0)
    );

    std::cout << "Coarse operator output norm = " << std::sqrt(coarse_outout_squared) << "\n";


    // // construct b = A*u_exact
    // A(b_device, u_device);

    // 0 initial guess
    DeviceVector x_device(V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));

    // create CG solver
    elasticity::CGSolver<DeviceVector> cg_solver(V->dofmap()->index_map, 3);
    cg_solver.set_max_iterations(2000);
    cg_solver.set_tolerance(T(1e-8));

    DeviceVector diagonal_inverse(V->dofmap()->index_map, 3);
    thrust::fill(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(), T(0));
    
    assemble_elasticity_diagonal<polynomial_order>(diagonal_inverse, phi_data, K, wdetJ,
      gpu_dofmap.map(), cell_list, bc_marker_device);
    
    // convert diag(A) to diag(A)^{-1}
    thrust::transform(thrust::device, diagonal_inverse.array().begin(), diagonal_inverse.array().end(),
      bc_marker_device.begin(), diagonal_inverse.array().begin(), invert_jacobi_diagonal<T>());

    if (jacobi_cg){
      // copy diag(A)^{-1} to CG solver
      cg_solver.set_diag_inverse(diagonal_inverse);
    }

    device_synchronize();


    // testing jacobi smoother
    DeviceVector x_smoother_test(V->dofmap()->index_map, 3);
    DeviceVector fine_Ax(V->dofmap()->index_map, 3);
    DeviceVector fine_residual(V->dofmap()->index_map, 3);

    thrust::fill(thrust::device, x_smoother_test.array().begin(), x_smoother_test.array().end(), T(0));

    const T residual_before = residual_norm(A, x_smoother_test, b_device, fine_Ax, fine_residual);
    std::cout << "Residual norm before Jacobi smoothing = " << residual_before << "\n";

    constexpr int jacobi_steps = 1;
    constexpr T omega = T(0.3);

    jacobi_smooth(A, x_smoother_test, b_device, diagonal_inverse, fine_Ax, fine_residual, jacobi_steps, omega);

    const T residual_after = residual_norm(A, x_smoother_test, b_device, fine_Ax, fine_residual);
    std::cout << "Residual norm after Jacobi smoothing = " << residual_after << "\n";


    // timing 
    constexpr int runs = 1;
    std::vector<double> times;
    times.reserve(runs);

    int iterations = 0;

    for (int i = 0; i < runs; ++i){
      thrust::fill(thrust::device, x_device.array().begin(), x_device.array().end(), T(0));
      device_synchronize();

      const auto start = std::chrono::high_resolution_clock::now();

      iterations = cg_solver.solve(A, x_device, b_device, jacobi_cg);
      device_synchronize();

      const auto end = std::chrono::high_resolution_clock::now();

      const double seconds = std::chrono::duration<double>(end - start).count();
      times.push_back(seconds);
    }

    const double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    std::cout << "Average solve time = " << avg_time << " seconds\n";
    std::cout << "Number of iterations = " << iterations << "\n";


    auto x = std::make_shared<fem::Function<T>>(V);

    thrust::copy(x_device.array().begin(), x_device.array().end(), x->x()->array().begin());
    thrust::copy(b_device.array().begin(), b_device.array().end(), b->x()->array().begin());

    std::cout << std::setprecision(17);

    // std::cout << "Exact u norm = "
    //           << dolfinx::la::norm(*u->x()) << "\n";

    std::cout << "Computed x norm = "
              << dolfinx::la::norm(*x->x()) << "\n";

    std::cout << "b norm = "
              << dolfinx::la::norm(*b->x()) << "\n";

              
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

    // number of global fine nodes
    const std::size_t num_fine_nodes = fine_from_coarse.array().size() / 3;

    // fine-space cell dofmap on CPU
    const auto fine_dofmap_host = V->dofmap()->map();

    // for every global fine node, we need to store:
    // 1. one mesh cell containing that fine node (the "owner" cell)
    // 2. the local node number of the fine node within that mesh cell
    std::vector<std::int32_t> owner_cell_host(num_fine_nodes, -1);
    std::vector<std::int32_t> owner_local_host(num_fine_nodes, -1);

    for (std::size_t cell = 0; cell < fine_dofmap_host.extent(0); ++cell)
    {
      for (int fine_i = 0; fine_i < fine_dofs_kernel; ++fine_i)
      {
        const std::int32_t fine_node = fine_dofmap_host(cell, fine_i);
        // if this fine node has not yet been assigned an owner cell, assign it now
        if (owner_cell_host[fine_node] == -1){
          owner_cell_host[fine_node] = static_cast<std::int32_t>(cell);
          owner_local_host[fine_node] = fine_i;
        }
      }
    }

    // check that every fine node recieved an owner
    for (std::size_t fine_node = 0; fine_node < num_fine_nodes; ++fine_node){
      const std::int32_t cell = owner_cell_host[fine_node];
      const std::int32_t local = owner_local_host[fine_node];
      if (cell == -1 || local == -1){
        throw std::runtime_error("Fine node " + std::to_string(fine_node) + " has no owner cell");
      }
      if (fine_dofmap_host(cell, local) != static_cast<std::int32_t>(fine_node)){
        throw std::runtime_error("Fine node " + std::to_string(fine_node) + " is not owned by cell " + std::to_string(cell) + " at local index " + std::to_string(local));
      }
    }

    thrust::device_vector<std::int32_t> owner_cell_device(owner_cell_host.begin(), owner_cell_host.end());
    thrust::device_vector<std::int32_t> owner_local_device(owner_local_host.begin(), owner_local_host.end());


    // // testing prolongation //
    // const std::size_t num_cells = V->dofmap()->map().extent(0);
    // constexpr int transfer_threads = 256;
    // const std::size_t total_transfer_tasks = num_cells * fine_dofs_kernel * 3;
    // const std::size_t transfer_blocks = (total_transfer_tasks + transfer_threads - 1) / transfer_threads;

    // thrust::fill(thrust::device, fine_from_coarse.array().begin(), fine_from_coarse.array().end(), T(0));

    // p_transfer::prolong_cells<T, coarse_dofs_kernel, fine_dofs_kernel><<<static_cast<int>(transfer_blocks), transfer_threads>>>(
    //   num_cells,
    //   P_device.data().get(),
    //   gpu_dofmap_coarse.map().data().get(),
    //   gpu_dofmap.map().data().get(),
    //   coarse_device.array().data().get(),
    //   fine_from_coarse.array().data().get()
    // );

    // device_synchronize();

    // // get the exact values of the fine function
    // const auto fine_exact_values = u_fine->x()->array();

    // // copy the GPU result to host for comparison
    // std::vector<T> fine_from_coarse_host(fine_from_coarse.array().size());
    // thrust::copy(fine_from_coarse.array().begin(), fine_from_coarse.array().end(), fine_from_coarse_host.begin());

    // constexpr T tolerance = T(1e-12);
    // T max_error = T(0);
    // std::size_t failed_values = 0;

    // // loop over fine dofs and compare the computed values from the coarse function to the exact values from the fine function
    // for (std::size_t dof = 0; dof < fine_from_coarse_host.size(); ++dof)
    // {
    //   const T error = std::abs(fine_from_coarse_host[dof] - fine_exact_values[dof]);
    //   max_error = std::max(max_error, error);

    //   if (error > tolerance) failed_values++;
    // }

    // std::cout << std::scientific
    //           << std::setprecision(3)
    //           << "Maximum global prolongation error = "
    //           << max_error << '\n';

    // std::cout << "Number of failed values = "
    //           << failed_values << '\n';


    // // testing restriction //
    // DeviceVector coarse_restricted(V_coarse->dofmap()->index_map, 3);
    // thrust::fill(thrust::device, coarse_restricted.array().begin(), coarse_restricted.array().end(), T(0));

    // const std::size_t total_restrict_tasks = num_fine_nodes * 3;
    // const std::size_t restrict_blocks = (total_restrict_tasks + transfer_threads - 1) / transfer_threads;

    // p_transfer::restrict_nodes<T, coarse_dofs_kernel, fine_dofs_kernel><<<static_cast<int>(restrict_blocks), transfer_threads>>>(
    //   num_fine_nodes,
    //   P_device.data().get(),
    //   gpu_dofmap_coarse.map().data().get(),
    //   owner_cell_device.data().get(),
    //   owner_local_device.data().get(),
    //   fine_from_coarse.array().data().get(),
    //   coarse_restricted.array().data().get()
    // );

    // device_synchronize();

    // const T left = thrust::inner_product(thrust::device, fine_from_coarse.array().begin(), fine_from_coarse.array().end(),
    //   fine_from_coarse.array().begin(), T(0));
    // const T right = thrust::inner_product(thrust::device, coarse_device.array().begin(), coarse_device.array().end(),
    //   coarse_restricted.array().begin(), T(0));

    // const T difference = std::abs(left - right);
    // const T scale = std::max(std::max(T(1), std::abs(left)), std::abs(right));
    // const T relative_error = difference / scale;

    // std::cout << std::setprecision(8);
    // std::cout << "(Px, Px) = " << left << "\n";
    // std::cout << "(x, P^T Px) = " << right << "\n";
    // std::cout << "Relative error = " << relative_error << "\n";

    // ///// for visualisation in PARAVIEW /////
    // // dolfinx function to hold the prolonged values for visualisation
    // auto u_prolonged = std::make_shared<fem::Function<T>>(V);
    // // copy GPU result into dolfinx function
    // std::copy(fine_from_coarse_host.begin(), fine_from_coarse_host.end(), u_prolonged->x()->array().begin());
    
    // u_fine->name = "direct_fine";
    // u_prolonged->name = "prolonged_fine";
    
    // #ifdef HAS_ADIOS2
    // io::VTXWriter<U> transfer_writer(MPI_COMM_WORLD, "p_transfer.bp", {u_fine, u_prolonged}, "bp4");
    // transfer_writer.write(0.0);
    // #endif

    // x->name = "displacement";

    // #ifdef HAS_ADIOS2
    // io::VTXWriter<U> solution_writer(MPI_COMM_WORLD, "cantilever.bp", {x}, "bp4");

    // solution_writer.write(0.0);
    // #endif
  }

  MPI_Finalize();
  return 0;
}
