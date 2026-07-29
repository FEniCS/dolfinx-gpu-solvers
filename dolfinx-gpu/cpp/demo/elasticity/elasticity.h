// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdint>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace detail
{
/// @brief Assemble action of elasticity operator (matrix-free)
///
/// Launch with blockDim = (cells_per_block, 3, ndofs), gridDim.x =
/// ceil(ncells / cells_per_block).
/// Thread utilisation per phase:
///   - Load:     30*cells_per_block threads
///   - Gradient+stress: nq*cells_per_block threads (q = ty*ndofs + tz < nq;
///     requires nq <= 3*ndofs)
///   - Scatter:  30*cells_per_block threads, no intra-block atomics
///
/// @param u Input degrees of freedom (global, node-interleaved)
/// @param b Output vector (accumulated via atomic adds)
/// @param phi_data Basis function values and reference gradients, layout:
///        [phi(nq*ndofs) | dphi/dxi(nq*ndofs) | dphi/deta(nq*ndofs) |
///        dphi/dzeta(nq*ndofs)] — constants of the reference element
/// @param K_entity Precomputed K = J^{-T} per cell per quadrature point,
///        layout: cell * 9*nq + component(0..8) * nq + q
/// @param wdetJ_entity Quadrature weight * |det J| per cell per quad point,
///        layout: cell * nq + q
/// @param cell_dofs Node indices per cell (not individual dof indices),
///        layout: cell * ndofs + local_node; global dof = node * 3 + component
/// @param cells Active cell indices, length ncells
/// @param ncells Number of active cells
/// @param lambda Lamé parameter λ
/// @param mu Lamé parameter μ
/// @tparam T               Scalar type (float or double)
/// @tparam nq              Number of quadrature points (e.g. 4 for affine P2
///                         tet, 14 for isoparametric)
/// @tparam ndofs           Number of scalar dofs per cell (e.g. 10 for P2 tet)
/// @tparam cells_per_block Number of cells processed per CUDA block; increase
///                         to raise occupancy (2 gives 2 warps per block)

template <typename T, int nq, int ndofs, int cells_per_block>

__global__ void elasticity_action(T* __restrict__ b, const T* __restrict__ u,
                                  const T* __restrict__ phi_data,
                                  const T* __restrict__ K_entity,
                                  const T* __restrict__ wdetJ_entity,
                                  const std::int32_t* __restrict__ cell_dofs,
                                  const std::int32_t* __restrict__ cells,
                                  int ncells,
                                  const std::int8_t* __restrict__ bc_marker,
                                  T lambda, 
                                  T mu)
{
  const int tx = threadIdx.x; // 0..cells_per_block-1 (cell within block)
  const int tz = threadIdx.z; // 0..ndofs-1           (node index)
  const int ty = threadIdx.y; // 0..2                 (component)

  // All threads must reach the __syncthreads() barriers below, so inactive
  // threads (cell_idx >= ncells in the last block) may not return early.
  // They use cell 0 as a placeholder for pointer arithmetic only; every
  // global/shared memory access is guarded by `active`.
  const int cell_idx = blockIdx.x * cells_per_block + tx;
  const bool active = cell_idx < ncells;
  const std::size_t cell_id = active ? static_cast<std::size_t>(cells[cell_idx]) : 0;

  // Shared memory: one scratch and wF slot per cell in the block
  __shared__ T scratch[3 * ndofs][cells_per_block]; // interleaved: [node*3 + component]
  __shared__ T wF[nq][3][3][cells_per_block]; // weighted flux tensor
  __shared__ std::int32_t nodes[ndofs][cells_per_block]; // node indices for each cell

  // --- Load dofs (all threads) ---
  if (active and ty ==0 and tz < ndofs) // global node number the same for all components so only one component loads the node indices
  {
    nodes[tz][tx] = cell_dofs[cell_id * ndofs + tz];
  }
  __syncthreads();

  if (active and tz < ndofs)
  {
    const std::int32_t node = nodes[tz][tx]; // global node number
    const std::int32_t dof = node* 3 + ty; // global dof number

    if (bc_marker[dof]) // check if node is clamped
    {
      scratch[tz * 3 + ty][tx] = T(0); // set the value to zero if it is clamped
    }
    else
    {
      scratch[tz * 3 + ty][tx] = u[dof]; // otherwise, load the value from the input vector
    }
  }

  __syncthreads();

  // Basis derivative arrays (phi values at offset 0 not needed)
  const T* dphix = phi_data + nq * ndofs;
  const T* dphiy = dphix + nq * ndofs;
  const T* dphiz = dphiy + nq * ndofs;

  // Geometry for this cell
  const T* K = K_entity + cell_id * 9 * nq;
  const T* wdetJ = wdetJ_entity + cell_id * nq;

  // --- Gradient + stress + out-transform (nq * cells_per_block threads) ---
  const int q = ty * ndofs + tz;
  if (active and q < nq)
  {
    // K = J^{-T}, stored component-major: K[k * nq + q]
    const T K0 = K[0 * nq + q], K1 = K[1 * nq + q], K2 = K[2 * nq + q];
    const T K3 = K[3 * nq + q], K4 = K[4 * nq + q], K5 = K[5 * nq + q];
    const T K6 = K[6 * nq + q], K7 = K[7 * nq + q], K8 = K[8 * nq + q];
    const T wJ = wdetJ[q];

    // Reference gradients: d(u_c)/d(xi_m) = sum_i scratch[i*3+c][tx] *
    // dphi_m[q*ndofs+i]
    T dux_xi = T(0), dux_eta = T(0), dux_zeta = T(0);
    T duy_xi = T(0), duy_eta = T(0), duy_zeta = T(0);
    T duz_xi = T(0), duz_eta = T(0), duz_zeta = T(0);

    for (int i = 0; i < ndofs; ++i)
    {
      const T bxi = dphix[q * ndofs + i];
      const T beta = dphiy[q * ndofs + i];
      const T bzeta = dphiz[q * ndofs + i];
      const T ux = scratch[i * 3 + 0][tx];
      const T uy = scratch[i * 3 + 1][tx];
      const T uz = scratch[i * 3 + 2][tx];

      dux_xi += ux * bxi;
      dux_eta += ux * beta;
      dux_zeta += ux * bzeta;
      duy_xi += uy * bxi;
      duy_eta += uy * beta;
      duy_zeta += uy * bzeta;
      duz_xi += uz * bxi;
      duz_eta += uz * beta;
      duz_zeta += uz * bzeta;
    }

    // Physical gradient: grad_x u = K * grad_xi u  (K = J^{-T})
    const T dux_dx = K0 * dux_xi + K1 * dux_eta + K2 * dux_zeta;
    const T dux_dy = K3 * dux_xi + K4 * dux_eta + K5 * dux_zeta;
    const T dux_dz = K6 * dux_xi + K7 * dux_eta + K8 * dux_zeta;

    const T duy_dx = K0 * duy_xi + K1 * duy_eta + K2 * duy_zeta;
    const T duy_dy = K3 * duy_xi + K4 * duy_eta + K5 * duy_zeta;
    const T duy_dz = K6 * duy_xi + K7 * duy_eta + K8 * duy_zeta;

    const T duz_dx = K0 * duz_xi + K1 * duz_eta + K2 * duz_zeta;
    const T duz_dy = K3 * duz_xi + K4 * duz_eta + K5 * duz_zeta;
    const T duz_dz = K6 * duz_xi + K7 * duz_eta + K8 * duz_zeta;

    // Stress: sigma = 2*mu*eps(u) + lambda*div(u)*I
    const T div_u = dux_dx + duy_dy + duz_dz;
    const T lam_div = lambda * div_u;
    const T sig_xx = T(2) * mu * dux_dx + lam_div;
    const T sig_yy = T(2) * mu * duy_dy + lam_div;
    const T sig_zz = T(2) * mu * duz_dz + lam_div;
    const T sig_xy = mu * (dux_dy + duy_dx);
    const T sig_xz = mu * (dux_dz + duz_dx);
    const T sig_yz = mu * (duy_dz + duz_dy);

    // Out-transform: wF[tx][q][i][m] = wJ * sum_j sigma_{ij} * K_{jm}
    wF[q][0][0][tx] = wJ * (sig_xx * K0 + sig_xy * K3 + sig_xz * K6);
    wF[q][0][1][tx] = wJ * (sig_xx * K1 + sig_xy * K4 + sig_xz * K7);
    wF[q][0][2][tx] = wJ * (sig_xx * K2 + sig_xy * K5 + sig_xz * K8);
    wF[q][1][0][tx] = wJ * (sig_xy * K0 + sig_yy * K3 + sig_yz * K6);
    wF[q][1][1][tx] = wJ * (sig_xy * K1 + sig_yy * K4 + sig_yz * K7);
    wF[q][1][2][tx] = wJ * (sig_xy * K2 + sig_yy * K5 + sig_yz * K8);
    wF[q][2][0][tx] = wJ * (sig_xz * K0 + sig_yz * K3 + sig_zz * K6);
    wF[q][2][1][tx] = wJ * (sig_xz * K1 + sig_yz * K4 + sig_zz * K7);
    wF[q][2][2][tx] = wJ * (sig_xz * K2 + sig_yz * K5 + sig_zz * K8);
  }

  __syncthreads();

  // --- Scatter (all threads, no intra-block atomics) ---
  // Thread (tz, ty, tx) uniquely owns (node=tz, component=ty) for its cell tx,
  // accumulates over quad points in registers, then one global atomic.
  if (active and tz < ndofs)
  {
    T contrib = T(0);
    for (int q = 0; q < nq; ++q)
    {
      const T dphi_xi = dphix[q * ndofs + tz];
      const T dphi_eta = dphiy[q * ndofs + tz];
      const T dphi_zeta = dphiz[q * ndofs + tz];
      contrib += wF[q][ty][0][tx] * dphi_xi + wF[q][ty][1][tx] * dphi_eta
                 + wF[q][ty][2][tx] * dphi_zeta;
    }
    const std::int32_t node = nodes[tz][tx]; // global node number
    const std::int32_t dof = node * 3 + ty; // global dof number

    if (!bc_marker[dof]) // only add to output if node is not clamped
    {
      atomicAdd(&b[dof], contrib); // add cell contribution to global output vector
    }
  }
}

// diagonal needed for Jacobi preconditioner
template <typename T, int nq, int ndofs, int cells_per_block>
__global__ void elasticity_diagonal(
  T* __restrict__ diagonal, // output diagonal vector
  const T* __restrict__ phi_data,
  const T* __restrict__ K_entity,
  const T* __restrict__ wdetJ_entity,
  const std::int32_t* __restrict__ cell_dofs,
  const std::int32_t* __restrict__ cells,
  int ncells,
  const std::int8_t* __restrict__ bc_marker,
  T lambda, 
  T mu){
    const int tx = threadIdx.x; // cell within block
    const int ty = threadIdx.y; // displacement component (0,1,2)
    const int tz = threadIdx.z; // local scalar basis function

    const int cell_idx = blockIdx.x * cells_per_block + tx;

    if (cell_idx >= ncells) return;

    const std::size_t cell_id = static_cast<std::size_t>(cells[cell_idx]);
    const std::int32_t node = cell_dofs[cell_id * ndofs + tz]; // global node number
    const std::int32_t dof = node * 3 + ty; // global scalar dof

    if (bc_marker[dof]) return; // skip clamped dofs, they're handled separately

    // reference basis derivatives
    const T* dphi_dxi = phi_data + nq * ndofs;
    const T* dphi_deta = dphi_dxi + nq * ndofs;
    const T* dphi_dzeta = dphi_deta + nq * ndofs;

    // geometry for this cell
    const T* K = K_entity + cell_id * 9 * nq;
    const T* wdetJ = wdetJ_entity + cell_id * nq;

    T cell_diagonal = T(0); // local diagonal contribution for this dof

    for (int q = 0; q < nq; ++q){
      // read reference basis derivatives for this quadrature point and dof
      const T dxi = dphi_dxi[q * ndofs + tz];
      const T deta = dphi_deta[q * ndofs + tz];
      const T dzeta = dphi_dzeta[q * ndofs + tz];

      // transform reference derivatives to physical derivatives using K = J^{-T}
      const T gx = K[0 * nq + q] * dxi + K[1 * nq + q] * deta + K[2 * nq + q] * dzeta;
      const T gy = K[3 * nq + q] * dxi + K[4 * nq + q] * deta + K[5 * nq + q] * dzeta;
      const T gz = K[6 * nq + q] * dxi + K[7 * nq + q] * deta + K[8 * nq + q] * dzeta;

      // form diagonal contribution
      const T gradient_squared = gx * gx + gy * gy + gz * gz;

      T component_gradient; // variable to hold derivative of the component corresponding to this thread

      if (ty == 0) component_gradient = gx;
      else if (ty == 1) component_gradient = gy;
      else component_gradient = gz;

      cell_diagonal += wdetJ[q] * (mu * gradient_squared + (lambda + mu) * component_gradient * component_gradient);
    }

    atomicAdd(&diagonal[dof], cell_diagonal);
  }

  // // eventually could implement GPU kernel for right hand side, but for now we'll just use the CPU dolfinx version
  // template <typename T, int nq, int dofs, int cells_per_block>
  // __global__ void elasticity_body_force(
  //   T* __restrict__ b,
  //   const T* __restrict__ phi_data,
  //   const T* __restrict__ wdetJ_entity,
  //   const std::int32_t* __restrict__ cell_dofs,
  //   const std::int32_t* __restrict__ cells,
  //   int ncells,
  //   const std::int8_t* __restrict__ bc_marker,
  //   T force_x, T force_y, T force_z
  // ){
  // }

  template <int P>
  struct elasticity_traits;

  template <>
  struct elasticity_traits<1>{
    static constexpr int ndofs = 4; // number of scalar dofs per cell
    static constexpr int quadrature_degree = 1; // quadrature degree for P1 tetrahedra
    static constexpr int nq = 1;     // number of quadrature points per cell
    #if defined(__HIP_PLATFORM_AMD__)
      static constexpr int cells_per_block = 16;
    #else
      static constexpr int cells_per_block = 32; // number of cells per CUDA block
      #endif
  };

  template <>
  struct elasticity_traits<2>{
    static constexpr int ndofs = 10; // number of scalar dofs per cell
    static constexpr int quadrature_degree = 2; // quadrature degree for P2 tetrahedra
    static constexpr int nq = 4;     // number of quadrature points per cell
    #if defined(__HIP_PLATFORM_AMD__)
      static constexpr int cells_per_block = 16;
    #else
      static constexpr int cells_per_block = 32; // number of cells per CUDA block
      #endif
  };

  template <>
  struct elasticity_traits<3>{
    static constexpr int ndofs = 20; // number of scalar dofs per cell
    static constexpr int quadrature_degree = 4; // quadrature degree for P3 tetrahedra
    static constexpr int nq = 14;     // number of quadrature points per cell
    #if defined(__HIP_PLATFORM_AMD__)
      static constexpr int cells_per_block = 8;
    #else
      static constexpr int cells_per_block = 16; // number of cells per CUDA block
    #endif
  };

  template <typename T>
  __global__ void set_identity_rows(T* output, const T* input, const std::int8_t* bc_marker, std::size_t size){
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size and bc_marker[i]){
      output[i] = input[i]; // set the value to the input if it is clamped
    }
  }
} // namespace detail


/// @brief Assemble 3D elasticity action vector
/// @param phi_data Basis evaluation data at reference quadrature points
/// @param K=J^{-T} Geometry transform at each quadrature point
/// @param wdetJ Weighted geometry detJ at each quadrature point
/// @param cell_dofs DofMap
/// @param cells List of cells to integrate over

template <int P, typename T, typename ContainerT, typename ContainerI, typename ContainerB>

void assemble_elasticity_action(dolfinx::la::Vector<T, ContainerT>& b,
                                const dolfinx::la::Vector<T, ContainerT>& u,
                                const ContainerT& phi_data, 
                                const ContainerT& K,
                                const ContainerT& wdetJ,
                                const ContainerI& cell_dofs,
                                const ContainerI& cells,
                                const ContainerB& bc_marker,
                                const T lambda = T(1), 
                                const T mu = T(1))
{
  constexpr int ndofs = detail::elasticity_traits<P>::ndofs;
  constexpr int nq = detail::elasticity_traits<P>::nq;
  constexpr int cells_per_block = detail::elasticity_traits<P>::cells_per_block;

  dim3 block_size(cells_per_block, 3, ndofs);
  dim3 grid_size((cells.size() + cells_per_block - 1) / cells_per_block);

  detail::elasticity_action<T, nq, ndofs, cells_per_block><<<grid_size, block_size>>>(
      b.array().data().get(), u.array().data().get(), phi_data.data().get(),
      K.data().get(), wdetJ.data().get(), cell_dofs.data().get(),
      cells.data().get(), cells.size(), bc_marker.data().get(), lambda, mu);

  constexpr int identity_threads = 256;
  const std::size_t identity_blocks = (b.array().size() + identity_threads - 1) / identity_threads;
  
  detail::set_identity_rows<T><<<static_cast<unsigned int>(identity_blocks), identity_threads>>>(
      b.array().data().get(), u.array().data().get(), bc_marker.data().get(), b.array().size());
}


template <int P, typename T, typename ContainerT, typename ContainerI, typename ContainerB>

void assemble_elasticity_diagonal(dolfinx::la::Vector<T, ContainerT>& diagonal,
                                  const ContainerT& phi_data, 
                                  const ContainerT& K,
                                  const ContainerT& wdetJ,
                                  const ContainerI& cell_dofs,
                                  const ContainerI& cells,
                                  const ContainerB& bc_marker,
                                  const T lambda = T(1), 
                                  const T mu = T(1))
{
  constexpr int ndofs = detail::elasticity_traits<P>::ndofs;
  constexpr int nq = detail::elasticity_traits<P>::nq;
  constexpr int cells_per_block = detail::elasticity_traits<P>::cells_per_block;

  dim3 block_size(cells_per_block, 3, ndofs);
  dim3 grid_size((cells.size() + cells_per_block - 1) / cells_per_block);

  detail::elasticity_diagonal<T, nq, ndofs, cells_per_block><<<grid_size, block_size>>>(
      diagonal.array().data().get(), phi_data.data().get(),
      K.data().get(), wdetJ.data().get(), cell_dofs.data().get(),
      cells.data().get(), cells.size(), bc_marker.data().get(), lambda, mu);
}