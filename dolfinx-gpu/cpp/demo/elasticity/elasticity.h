// Copyright (C) 2026 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <cstdint>

namespace detail
{
/// @brief Assemble action of elasticity operator (matrix-free)
///
/// Launch with blockDim = (ndofs, 3, cells_per_block), gridDim.x =
/// ceil(ncells / cells_per_block).
/// Thread utilisation per phase:
///   - Load:     30*cells_per_block threads
///   - Gradient+stress: nq*cells_per_block threads (q = ty*ndofs + tx < nq;
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
                                  int ncells, T lambda, T mu)
{
  const int tz = threadIdx.z; // 0..cells_per_block-1 (cell within block)
  const int tx = threadIdx.x; // 0..ndofs-1           (node index)
  const int ty = threadIdx.y; // 0..2                 (component)

  // All threads must reach the __syncthreads() barriers below, so inactive
  // threads (cell_idx >= ncells in the last block) may not return early.
  // They use cell 0 as a placeholder for pointer arithmetic only; every
  // global/shared memory access is guarded by `active`.
  const int cell_idx = blockIdx.x * cells_per_block + tz;
  const bool active = cell_idx < ncells;
  const std::size_t cell_id
      = active ? static_cast<std::size_t>(cells[cell_idx]) : 0;

  // Shared memory: one scratch and wF slot per cell in the block
  __shared__ T
      scratch[cells_per_block][3 * ndofs]; // interleaved: [node*3 + component]
  __shared__ T wF[cells_per_block][nq][3][3]; // weighted flux tensor

  // --- Load dofs (all threads) ---
  if (active and tx < ndofs)
  {
    const std::int32_t dof = cell_dofs[cell_id * ndofs + tx] * 3 + ty;
    scratch[tz][tx * 3 + ty] = u[dof];
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
  const int q = ty * ndofs + tx;
  if (active and q < nq)
  {
    // K = J^{-T}, stored component-major: K[k * nq + q]
    const T K0 = K[0 * nq + q], K1 = K[1 * nq + q], K2 = K[2 * nq + q];
    const T K3 = K[3 * nq + q], K4 = K[4 * nq + q], K5 = K[5 * nq + q];
    const T K6 = K[6 * nq + q], K7 = K[7 * nq + q], K8 = K[8 * nq + q];
    const T wJ = wdetJ[q];

    // Reference gradients: d(u_c)/d(xi_m) = sum_i scratch[tz][i*3+c] *
    // dphi_m[q*ndofs+i]
    T dux_xi = T(0), dux_eta = T(0), dux_zeta = T(0);
    T duy_xi = T(0), duy_eta = T(0), duy_zeta = T(0);
    T duz_xi = T(0), duz_eta = T(0), duz_zeta = T(0);

    for (int i = 0; i < ndofs; ++i)
    {
      const T bxi = dphix[q * ndofs + i];
      const T beta = dphiy[q * ndofs + i];
      const T bzeta = dphiz[q * ndofs + i];
      const T ux = scratch[tz][i * 3 + 0];
      const T uy = scratch[tz][i * 3 + 1];
      const T uz = scratch[tz][i * 3 + 2];

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

    // Out-transform: wF[tz][q][i][m] = wJ * sum_j sigma_{ij} * K_{jm}
    wF[tz][q][0][0] = wJ * (sig_xx * K0 + sig_xy * K3 + sig_xz * K6);
    wF[tz][q][0][1] = wJ * (sig_xx * K1 + sig_xy * K4 + sig_xz * K7);
    wF[tz][q][0][2] = wJ * (sig_xx * K2 + sig_xy * K5 + sig_xz * K8);
    wF[tz][q][1][0] = wJ * (sig_xy * K0 + sig_yy * K3 + sig_yz * K6);
    wF[tz][q][1][1] = wJ * (sig_xy * K1 + sig_yy * K4 + sig_yz * K7);
    wF[tz][q][1][2] = wJ * (sig_xy * K2 + sig_yy * K5 + sig_yz * K8);
    wF[tz][q][2][0] = wJ * (sig_xz * K0 + sig_yz * K3 + sig_zz * K6);
    wF[tz][q][2][1] = wJ * (sig_xz * K1 + sig_yz * K4 + sig_zz * K7);
    wF[tz][q][2][2] = wJ * (sig_xz * K2 + sig_yz * K5 + sig_zz * K8);
  }

  __syncthreads();

  // --- Scatter (all threads, no intra-block atomics) ---
  // Thread (tx, ty, tz) uniquely owns (node=tx, component=ty) for its cell tz,
  // accumulates over quad points in registers, then one global atomic.
  if (active and tx < ndofs)
  {
    T contrib = T(0);
    for (int q = 0; q < nq; ++q)
    {
      const T dphi_xi = dphix[q * ndofs + tx];
      const T dphi_eta = dphiy[q * ndofs + tx];
      const T dphi_zeta = dphiz[q * ndofs + tx];
      contrib += wF[tz][q][ty][0] * dphi_xi + wF[tz][q][ty][1] * dphi_eta
                 + wF[tz][q][ty][2] * dphi_zeta;
    }
    const std::int32_t dof = cell_dofs[cell_id * ndofs + tx] * 3 + ty;
    atomicAdd(&b[dof], contrib);
  }
}

} // namespace detail

/// @brief Assemble 3D elasticity action vector
/// @param phi_data Basis evaluation data at reference quadrature points
/// @param K=adj(J) Geometry transform at each quadrature point
/// @param wdetJ Weighted geometry detJ at each quadrature point
/// @param cell_dofs DofMap
/// @param cells List of cells to integrate over
template <typename T, typename ContainerT, typename ContainerI>
void assemble_elasticity_action(dolfinx::la::Vector<T, ContainerT>& b,
                                const dolfinx::la::Vector<T, ContainerT>& u,
                                const ContainerT& phi_data, const ContainerT& K,
                                const ContainerT& wdetJ,
                                const ContainerI& cell_dofs,
                                const ContainerI& cells)
{
  constexpr int ndofs = 10;
  constexpr int nq = 4;

  dim3 block_size(ndofs, 3, 2);
  dim3 grid_size(cells.size() / 2 + 1);

  detail::elasticity_action<T, nq, ndofs, 2><<<grid_size, block_size>>>(
      b.array().data().get(), u.array().data().get(), phi_data.data().get(),
      K.data().get(), wdetJ.data().get(), cell_dofs.data().get(),
      cells.data().get(), cells.size());
}
