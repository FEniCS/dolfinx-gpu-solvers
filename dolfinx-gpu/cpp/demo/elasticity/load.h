// Copyright (C) 2026 Arwa Fathy
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <thrust/device_vector.h>

namespace detail
{
// for constant force, x y z componnts are diliberatly ignored
template <typename T>
struct ConstantBodyForce
{
  T force_x, force_y, force_z;

  __host__ __device__ T operator()(int component, T, T, T) const
  {
    if (component == 0)
      return force_x;
    else if (component == 1)
      return force_y;
    else
      return force_z;
  }
};

/// generic assembly kernel for force vector
///
/// @param b output vector
/// @param phi_data basis function values at quadrature points
/// @param wdetJ quadrature weights times determinant of Jacobian
/// @param dof_coordinates physical coordinates of the dofs
/// @param cell_dofs mapping from cell to global dofs
/// @param cells list of cells to process
/// @param bc_marker boundary condition marker for each dof
/// @param num_cells number of cells
/// @param force body force evaluator
template <typename T, int nq, int ndofs, typename ForceEvaluator>
__global__ void
body_force_kernel(T* __restrict__ b, const T* __restrict__ phi_data,
                  const T* __restrict__ wdetJ,
                  const T* __restrict__ dof_coordinates,
                  const std::int32_t* __restrict__ cell_dofs,
                  const std::int32_t* __restrict__ cells,
                  const std::int8_t* __restrict__ bc_marker,
                  const std::size_t num_cells, ForceEvaluator force)
{
  // calculate the global thread index
  const std::size_t task
      = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // number of tasks per cell
  constexpr int values_per_cell = 3 * ndofs; // 3 values per dof (x, y, z)
  const std::size_t total_tasks = num_cells * values_per_cell;

  if (task >= total_tasks)
    return; // exit if out of bounds

  // which cell
  const std::size_t cell_idx = task / values_per_cell;

  // get the actual cell index
  const std::size_t cell_id = static_cast<std::size_t>(cells[cell_idx]);

  // position inside cell
  const std::size_t local_task = (task % values_per_cell);

  // reverse the flattened ordering essentially

  // which component (x, y, z)
  const int component = static_cast<int>(local_task % 3);
  // local basis function
  const int local_node = local_task / 3;

  // initialise cell contribution i.e. local accumulator to hold the
  // contribution for this cell, local node and component
  T cell_value = T(0);

  for (int q = 0; q < nq; ++q) // loop over quadrature points
  {
    // basis function value at quadrature point q for local node
    const T phi_i = phi_data[q * ndofs + local_node];
    // weight times determinant of Jacobian for this cell and quadrature point
    const T wj = wdetJ[cell_id * nq + q];

    T xq = T(0);
    T yq = T(0);
    T zq = T(0);

    // interpolate node coordinates to quadrature point
    for (int i = 0; i < ndofs; ++i)
    {
      // get global node index
      const std::int32_t global_node = cell_dofs[cell_id * ndofs + i];

      // basis function value for this node at quadrature point q
      const T phi = phi_data[q * ndofs + i];

      xq += phi * dof_coordinates[3 * global_node + 0]; // x coordinate
      yq += phi * dof_coordinates[3 * global_node + 1]; // y coordinate
      zq += phi * dof_coordinates[3 * global_node + 2]; // z coordinate
    }
    // evaluate the force at this quadrature point
    const T force_value = force(component, xq, yq, zq);

    // accumulate contribution
    cell_value += phi_i * wj * force_value;
  }

  // get global dof index
  const std::int32_t global_node = cell_dofs[cell_id * ndofs + local_node];

  // global index in the output
  const std::size_t global_dof = 3 * global_node + component;

  // apply boundary condition and assemble
  if (!bc_marker[global_dof]) // if not a boundary condition node
  {
    atomicAdd(&b[global_dof], cell_value); // atomic add to global vector
  }
}
} // namespace detail

/// wrapper to launch kernel
///
/// @param b output vector
/// @param phi_data basis function values at quadrature points
/// @param wdetJ quadrature weights times determinant of Jacobian
/// @param dof_coordinates physical coordinates of the dofs
/// @param cell_dofs mapping from cell to global dofs
/// @param cells list of cells to process
/// @param bc_marker boundary condition marker for each dof
/// @param num_cells number of cells
/// @param force functor to evaluate the force at a given point
template <int P, typename DeviceVector, typename ScalarContainer,
          typename IndexContainer, typename BCMarkerContainer,
          typename ForceEvaluator>
void launch_body_force_kernel(DeviceVector& b, const ScalarContainer& phi_data,
                              const ScalarContainer& wdetJ,
                              const ScalarContainer& dof_coordinates,
                              const IndexContainer& cell_dofs,
                              const IndexContainer& cells,
                              const BCMarkerContainer& bc_marker,
                              ForceEvaluator force)
{
  // deduce the scalar type from the device vector
  using T = typename DeviceVector::value_type;

  // number of quadrature points for P-th order elements
  constexpr int nq = detail::elasticity_traits<P>::nq;

  // number of local degrees of freedom
  constexpr int ndofs = detail::elasticity_traits<P>::ndofs;

  // initialise output vector to zero
  thrust::fill(thrust::device, b.array().begin(), b.array().end(), T(0));

  // total number of tasks
  const std::size_t total_tasks = cells.size() * 3 * ndofs;

  // number of threads per block
  constexpr int block_size = 256;

  // number of blocks needed
  const int grid_size = (total_tasks + block_size - 1) / block_size;

  detail::body_force_kernel<T, nq, ndofs, ForceEvaluator>
      <<<grid_size, block_size>>>(
          b.array().data().get(), phi_data.data().get(), wdetJ.data().get(),
          dof_coordinates.data().get(), cell_dofs.data().get(),
          cells.data().get(), bc_marker.data().get(), cells.size(), force);
}
