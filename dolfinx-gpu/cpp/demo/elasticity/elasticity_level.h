#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/la/Vector.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "../../include/gpu_geometry.h"
#include "elasticity.h"
#include "load.h"

using T = double; // float or double
using U = dolfinx::scalar_value_t<T>;


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


// Traits for elasticity kernel parameters
template <int P>
class ElasticityLevel{
  public:
    using DeviceScalarVector = thrust::device_vector<T>;
    using DeviceIndexVector = thrust::device_vector<std::int32_t>;
    using DeviceMarkerVector = thrust::device_vector<std::int8_t>;
    using DeviceVector = dolfinx::la::Vector<T, DeviceScalarVector, DeviceIndexVector>;

    static constexpr int order = P;
    static constexpr int ndofs = detail::elasticity_traits<P>::ndofs;
    static constexpr int quadrature_degree = detail::elasticity_traits<P>::quadrature_degree;
  
  private:
    // all polynomial meshes use the same mesh cells
    const DeviceIndexVector& _cell_list;
    const T lambda_;
    const T mu_;
  
  public:
    basix::FiniteElement<U> elem;
    std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V;
    GPUDofMap<DeviceIndexVector> gpu_dofmap;
    GPUGeometry<thrust::device_vector<U>, DeviceIndexVector> geometry;

    std::size_t nq = 0;

    DeviceScalarVector phi_data;
    DeviceScalarVector K;
    DeviceScalarVector wdetJ;
    DeviceScalarVector dof_coordinates;
    DeviceMarkerVector bc_marker;
    std::vector<std::int32_t> bc_nodes;

    template <typename BoundaryLocator>
    ElasticityLevel(
      const std::shared_ptr<dolfinx::mesh::Mesh<U>>& mesh_ptr,
      const DeviceIndexVector& cell_list,
      const BoundaryLocator& boundary_locator,
      const T lambda_,
      const T mu_)
      : _cell_list(cell_list),
        lambda_(lambda_),
        mu_(mu_),
        elem(basix::create_element<U>(
          basix::element::family::P, basix::cell::type::tetrahedron, P,
          basix::element::lagrange_variant::equispaced,
          basix::element::dpc_variant::unset,
          /* discontinuous = */ false)),
        V(std::make_shared<dolfinx::fem::FunctionSpace<U>>(dolfinx::fem::create_functionspace<U>(
          mesh_ptr, std::make_shared<dolfinx::fem::FiniteElement<U>>(
            elem, std::vector<std::size_t>{3})))),
          gpu_dofmap(*(V->dofmap())),
          geometry(mesh_ptr->geometry(), quadrature_degree)
    {
      build_geometry();
      build_basis();
      build_dof_coordinates();
      build_bc_marker(boundary_locator);
    }

    // apply elasticity operator for this level
    void operator()(DeviceVector& output, DeviceVector& input) const
    {
      // update ghost input values
      scatter_fwd(input);

      // assemble local cell contibutions
      thrust::fill(thrust::device, output.array().begin(), output.array().end(), T(0));
      assemble_elasticity_action<P>(output, input, phi_data, K, wdetJ, gpu_dofmap.map(), _cell_list, bc_marker, lambda_, mu_);

      // accumulate ghost contributions to owning ranks
      scatter_rev_add(output);

      // apply Dirichlet boundary conditions
      const std::size_t owned_size = output.bs() * output.index_map()->size_local();

      constexpr int block_size = 256;
      const std::size_t num_blocks = (owned_size + block_size - 1) / block_size;

      detail::set_identity_rows<T><<<num_blocks, block_size>>>(
          output.array().data().get(), input.array().data().get(), bc_marker.data().get(), owned_size);
    }

    // assemble the diagonal of the elasticity operator for this level
    void assemble_diagonal(DeviceVector& diagonal) const
    {
      thrust::fill(thrust::device, diagonal.array().begin(), diagonal.array().end(), T(0));
      assemble_elasticity_diagonal<P>(diagonal, phi_data, K, wdetJ, gpu_dofmap.map(), _cell_list, bc_marker, lambda_, mu_);

      scatter_rev_add(diagonal);
    }

    // assemble the body force vector for this level
    template <typename ForceEvaluator>
    void assemble_body_force(DeviceVector& b, ForceEvaluator force) const
    {
      launch_body_force_kernel<P>(b, phi_data, wdetJ, dof_coordinates, gpu_dofmap.map(), _cell_list, bc_marker, force);

      scatter_rev_add(b);
    }

    template <typename ExactSolution>
    T compute_l2_error_squared(const DeviceVector& solution, ExactSolution exact_solution) const
    {
      return launch_l2_error_squared_kernel<P>(solution, phi_data, wdetJ, dof_coordinates, gpu_dofmap.map(), _cell_list, exact_solution);
    }
  
  private:
    void build_geometry(){
      // three coordinates per quadrature point
      nq = geometry.qpoints().size() / 3;

      wdetJ.resize(_cell_list.size() * nq);
      geometry.compute_detJ(wdetJ, _cell_list);

      K.resize(_cell_list.size() * nq * 9);
      geometry.compute_K9(K, wdetJ, _cell_list);
    }

    void build_basis(){
      auto shape = elem.tabulate_shape(1, nq);
      const std::size_t table_size = std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<std::size_t>());
      std::vector<T> table(table_size);
      std::vector<T> qpoints(geometry.qpoints().size());

      thrust::copy(geometry.qpoints().begin(), geometry.qpoints().end(), qpoints.begin());
      elem.tabulate(1, std::span(qpoints), {nq, 3}, std::span(table));

      phi_data = DeviceScalarVector(table.begin(), table.end());
    }

    template <typename FacetContainer>
    void build_bc_marker(const FacetContainer& boundary_facets){
      const int tdim = V->mesh()->topology()->dim();
      const int fdim = tdim - 1;

      V->mesh()->topology_mutable()->create_entities(fdim);
      V->mesh()->topology_mutable()->create_connectivity(fdim, tdim);
      V->mesh()->topology_mutable()->create_connectivity(tdim, fdim);

      // find the clamped nodes
      bc_nodes = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology(), 
        *V->dofmap(), 
        fdim, 
        std::span<const std::int32_t>(boundary_facets.data(), boundary_facets.size()));

      std::cout << "Number of Dirichlet boundary nodes: " << bc_nodes.size() << std::endl;

      // create a temporary function to obtain the scalar vector size
      auto size_function = std::make_shared<dolfinx::fem::Function<T>>(V);
      std::vector<std::int8_t> bc_marker_host(size_function->x()->array().size(), false);

      for (std::size_t node : bc_nodes){
        bc_marker_host[3 * node + 0] = true;
        bc_marker_host[3 * node + 1] = true;
        bc_marker_host[3 * node + 2] = true;
      }

      bc_marker.assign(bc_marker_host.begin(), bc_marker_host.end());
    }

    void build_dof_coordinates(){
      const std::vector<T> coordinates_host = V->tabulate_dof_coordinates(false); // asks dolfinx to get physical coordinates of the nodes in this function space
      dof_coordinates.assign(coordinates_host.begin(), coordinates_host.end()); // copy to device vector
    }
};