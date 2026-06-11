// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier: MIT
//

#include <basix/finite-element.h>
#include <dolfinx.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

#include "../../include/gpu_geometry.h"
#include "elasticity.h"

using namespace dolfinx;
using T = double;
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

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    int n = 2;
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
        g_device(mesh->geometry(), 4);
    thrust::device_vector<T> K(cell_list.size() * 9
                               * g_device.qpoints().size());
    g_device.compute_K9(K, cell_list);
    thrust::device_vector<T> wdetJ(cell_list.size()
                                   * g_device.qpoints().size());
    g_device.compute_detJ(wdetJ, cell_list);

    // -----------------------------------------------------------------------
    // Finite element space
    // -----------------------------------------------------------------------
    auto elem_p2 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 2,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset,
        /* discontinuous = */ false);

    auto V
        = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      elem_p2, std::vector<std::size_t>{3})));
    GPUDofMap<thrust::device_vector<std::int32_t>> gpu_dofmap(*(V->dofmap()));

    // -----------------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------------
    auto u = std::make_shared<fem::Function<T>>(V);

    // Copy u to device
    la::Vector<T, thrust::device_vector<T>> u_device(*(u->x()));
    la::Vector<T, thrust::device_vector<T>> b_device(V->dofmap()->index_map, 3);

    // TODO: interpolate something into u

    // TODO: fill in phi from basix
    thrust::device_vector<T> phi_data;

    assemble_elasticity_action(b_device, u_device, phi_data, K, wdetJ,
                               gpu_dofmap.map(), cell_list);

    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
