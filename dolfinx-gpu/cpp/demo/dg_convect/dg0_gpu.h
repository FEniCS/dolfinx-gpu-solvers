
// GPU kernels for explicit DG0 upwind convection on tetrahedral meshes.
//
// Pipeline (launch in order):
//   1. compute_w_at_qp  – evaluate P1 velocity w at one quadrature point per
//                         local facet, stored cell-by-cell
//   2. dg0_convection   – accumulate upwind flux into b using those values
//   3. dg0_mass         – apply the explicit Euler update to u_n in-place
//
// run_dg0_convection is the host wrapper that sequences all three.

#pragma once
#include <cstdint>
#include <thrust/device_vector.h>

// HIP requires the runtime header to be included explicitly for device
// built-ins (blockIdx, blockDim, threadIdx, atomicAdd, etc.).
// CUDA/nvcc injects these implicitly, so no include is needed there.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

// Evaluate the P1 DG velocity field w at one quadrature point per local facet
// of each cell, and store the results cell-by-cell.
//
// For cell cglobal and local facet f (0..3), the result is written to
//   w_q[(cglobal*4 + f)*3 + d]  for spatial component d.
//
// This layout matches the combined cell+local-facet index used by
// dg0_convection (c0f = cell << 2 | local_facet), so that each side of a
// shared facet carries its own independent w evaluation.
//
// @param w_dof   P1 DG velocity DOFs, layout [(cell*4 + node)*3 + component].
// @param phi     Basis function values at the facet quadrature points,
//                layout [local_facet*4 + node], size 16 (4 facets x 4 nodes).
//                For a P1 tet with one point per facet at the facet centroid:
//                phi[f*4+i] = 1/3 for i != f, 0 for i == f.
// @param w_q     Output, layout [(cell*4 + local_facet)*3 + component].
//                Must be zero-initialised before the kernel is launched.
// @param cells   List of cell indices to process.
// @param n_cells Length of cells.
template <typename T>
__global__ void compute_w_at_qp(const T* w_dof, const T* phi, T* w_q,
                                const std::int32_t* cells, int n_cells)
{
  // Load a set of cells
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;
  std::int32_t cglobal = cells[idx];

  // 4 basis functions on tet
  constexpr int nphi = 4;

  // Get facets of cell
  for (int f = 0; f < 4; ++f)
  {
    // Place to store facet qp data (each cell)
    int f0 = cglobal * 4 + f;
    for (int i = 0; i < nphi; ++i)
    {
      // DoF index
      int c0 = cglobal * nphi + i;
      T phi_val = phi[f * nphi + i];

      // Compute w at qp in cell
      w_q[f0 * 3] += w_dof[c0 * 3] * phi_val;
      w_q[f0 * 3 + 1] += w_dof[c0 * 3 + 1] * phi_val;
      w_q[f0 * 3 + 2] += w_dof[c0 * 3 + 2] * phi_val;
    }
  }
}

// Accumulate the upwind convective flux across each interior facet into b.
//
// For the two cells c0, c1 sharing a facet (c0 < c1, c0 owns the outward
// normal), the upwind flux is:
//   flux = max(w0.n, 0)*u_n[c0] + min(w1.n, 0)*u_n[c1]
//   b[c0] -= flux;  b[c1] += flux;
// where w0, w1 are the per-cell w values at the quadrature point on that
// facet (from compute_w_at_qp), accessed via the combined cell+local-facet
// index stored in facet_to_cell.
//
// @param b             RHS accumulator, length num_cells. Zero before launch.
// @param u_n           DG0 solution at the previous time step.
// @param w             Output of compute_w_at_qp, layout [(cell*4+lf)*3+d].
// @param phi           Unused (reserved for higher-order extension).
// @param normals       Scaled outward facet normals, layout [facet*3+d].
//                      Magnitude encodes the facet area; points away from c0.
// @param facet_to_cell Layout [facet*2]: packed {(c0<<2)|lf0, (c1<<2)|lf1},
//                      c0 < c1. Cell index = value >> 2; local facet = value & 3.
// @param facets        Interior facet global indices to process.
// @param n_facets      Length of facets.
template <typename T>
__global__ void dg0_convection(T* b, const T* u_n, const T* w, const T* phi,
                               const T* normals,
                               const std::int32_t* facet_to_cell,
                               const int* facets, int n_facets)
{
  // Load a set of facets
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_facets)
    return;
  int fglobal = facets[idx];

  // Facet-to-cell list must be pre-sorted on each facet so that c0 is always
  // lower index

  // Get combined cell+local_facet index
  std::int32_t c0f = facet_to_cell[fglobal * 2];
  std::int32_t c1f = facet_to_cell[fglobal * 2 + 1];
  // Extract local facet indices from facet_to_cell (stored in lower two bits)
  // if needed
  //  std::int32_t flocal_0 = c0f & 0x03;
  //  std::int32_t flocal_1 = c1f & 0x03;

  // Get cell indices for DG0
  std::int32_t c0 = c0f >> 2;
  std::int32_t c1 = c1f >> 2;

  // Compute w.n on both sides of facet
  const T* n = normals + fglobal * 3;
  const T* w0 = w + c0f * 3;
  const T* w1 = w + c1f * 3;
  T w0n = n[0] * w0[0] + n[1] * w0[1] + n[2] * w0[2];
  T w1n = n[0] * w1[0] + n[1] * w1[1] + n[2] * w1[2];

  // Apply upwinding
  T flux = fmax(w0n, 0) * u_n[c0] + fmin(w1n, 0) * u_n[c1];

  atomicAdd(&b[c0], -flux);
  atomicAdd(&b[c1], flux);
}

// Apply the explicit Euler time update in-place:
//   u_n[c] += b[c] * 6*dt / detJ[c]
//
// For DG0 the mass matrix is diagonal with M[c] = |T_c|/dt = detJ[c]/(6*dt),
// so the full update u_new = u_old + b/M reduces to the above.
// b contains only facet-flux contributions; the u_old*M/M = u_old term is
// handled analytically.
//
// @param u_n    In/out: updated in-place.
// @param dt     Time step size.
// @param b      Facet flux RHS from dg0_convection.
// @param detJ   Cell Jacobian determinants det(J) (not pre-divided by 6).
// @param cells  Cell indices to update.
// @param n_cells Length of cells.
template <typename T>
__global__ void dg0_mass(T* u_n, T dt, const T* b, const T* detJ,
                         const int* cells, int n_cells)
{
  // Load a set of cells
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;
  int cglobal = cells[idx];

  // b_total = (u_n * detJ / (6*dt)) + b_facet
  // M_diag  = detJ / (6*dt)
  // u_new   = b_total / M_diag = b_total * (6*dt) / detJ
  u_n[cglobal] = u_n[cglobal] + b[cglobal] * (T(6) * dt) / detJ[cglobal];
}

// Host wrapper: advance u_n by one explicit Euler convection step.
// Launches compute_w_at_qp, dg0_convection, and dg0_mass in sequence,
// with cudaDeviceSynchronize() between each launch.
//
// w_q is allocated internally as a temporary of size w.size() (one 3-vector
// per cell per local facet). If w is time-independent this could be hoisted.
template <typename ContainerT, typename ContainerI>
void run_dg0_convection(ContainerT& b, ContainerT& u_n, const ContainerT& w,
                        const ContainerT& phi, const ContainerT& normals,
                        const ContainerT& detJ, const ContainerI& facet_to_cell,
                        const ContainerI& facets, const ContainerI& cells,
                        double dt)
{
  using T = typename ContainerT::value_type;

  // Choose a good block size
  dim3 block_size(512);
  dim3 grid_size(facets.size() / block_size.x + 1);

  // If w is constant, could move this outside loop
  thrust::device_vector<T> w_q(w.size(), T(0));
  grid_size = dim3(cells.size() / block_size.x + 1);
  compute_w_at_qp<T><<<grid_size, block_size>>>(
      w.data().get(), phi.data().get(), w_q.data().get(), cells.data().get(),
      cells.size());

  cudaDeviceSynchronize();

  grid_size = dim3(facets.size() / block_size.x + 1);
  dg0_convection<T><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w_q.data().get(), phi.data().get(),
      normals.data().get(), facet_to_cell.data().get(), facets.data().get(),
      facets.size());

  cudaDeviceSynchronize();

  grid_size = dim3(cells.size() / block_size.x + 1);
  dg0_mass<T><<<grid_size, block_size>>>(u_n.data().get(), dt, b.data().get(),
                                         detJ.data().get(), cells.data().get(),
                                         cells.size());
}
