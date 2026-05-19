
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
// @param u_n           DG1 solution at the previous time step.
// @param w             Velocity w as a DG1 function
// @param phi           Basis function evaluated at quadrature points on facets
// @param normals       Scaled outward facet normals, layout [facet*3+d].
//                      Magnitude encodes the facet area; points away from c0.
// @param facet_to_cell Layout [facet*2]: packed {(c0<<2)|lf0, (c1<<2)|lf1},
//                      c0 < c1. Cell index = value >> 2; local facet = value
//                      & 3.
// @param facets        Interior facet global indices to process.
// @param n_facets      Length of facets.
template <typename T>
__global__ void dg1_convection(T* b, const T* u_n, const T* w, const T* phi,
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
  // Extract local facet indices and perms from facet_to_cell (stored in lower
  // eight bits)
  std::int8_t flocal_0 = c0f & 0x03;
  std::int8_t fperm_0 = (c0f >> 2) & 0x3F;
  std::int8_t flocal_1 = c1f & 0x03;
  std::int8_t fperm_1 = (c1f >> 2) & 0x3F;
  // Get cell indices
  std::int32_t c0 = c0f >> 8;
  std::int32_t c1 = c1f >> 8;

  // Facet permutations for 6 point quadrature
  constexpr std::int8_t qperm[36]
      = {0, 1, 2, 3, 4, 5, 2, 4, 0, 5, 1, 3, 3, 5, 1, 4, 0, 2,
         5, 3, 4, 1, 2, 0, 4, 2, 5, 0, 3, 1, 1, 0, 3, 2, 5, 4};
  const std::int8_t* qp0 = qperm + fperm_0 * 6;
  const std::int8_t* qp1 = qperm + fperm_1 * 6;

  // Number of DoFs per cell (DG1)
  constexpr int ndof = 4;
  // Number of quadrature points per facet
  constexpr int nq = 6;

  const T* n = normals + fglobal * 3;

  T flux[nq];
  for (int iq = 0; iq < nq; ++iq)
  {
    // Compute w.n at quadrature points on facet of cell0
    T w0n = 0;
    for (int i = 0; i < ndof; ++i)
    {
      const T* w0 = w + (c0 * ndof + i) * 3;
      T phi_i = phi[flocal_0 * ndof * nq + qp0[iq] * ndof + i];
      w0n += w0[0] * phi_i * n[0];
      w0n += w0[1] * phi_i * n[1];
      w0n += w0[2] * phi_i * n[2];
    }

    // Compute w.n at quadrature points on facet of cell1
    T w1n = 0;
    for (int i = 0; i < ndof; ++i)
    {
      const T* w1 = w + (c1 * ndof + i) * 3;
      T phi_i = phi[flocal_1 * ndof * nq + qp1[iq] * ndof + i];
      w1n += w1[0] * phi_i * n[0];
      w1n += w1[1] * phi_i * n[1];
      w1n += w1[2] * phi_i * n[2];
    }

    // Get u value at quadrature points on facet, cell0
    const T* u0 = u_n + c0 * ndof;
    T uf0 = 0;
    for (int i = 0; i < ndof; ++i)
      uf0 += u0[i] * phi[flocal_0 * ndof * nq + qp0[iq] * ndof + i];

    // Get u value at quadrature points on facet, cell1
    const T* u1 = u_n + c1 * ndof;
    T uf1 = 0;
    for (int i = 0; i < ndof; ++i)
      uf1 += u1[i] * phi[flocal_1 * ndof * nq + qp1[iq] * ndof + i];

    // Apply upwinding at quadrature points
    flux[iq] = fmax(w0n, 0) * uf0 + fmin(w1n, 0) * uf1;
  }

  // Locate b cell dofs in output data
  T* b0 = b + c0 * ndof;
  T* b1 = b + c1 * ndof;
  for (int i = 0; i < ndof; ++i)
  {
    T b0val = 0;
    T b1val = 0;
    for (int iq = 0; iq < nq; ++iq)
    {
      b0val -= phi[flocal_0 * ndof * nq + qp0[iq] * ndof + i] * flux[iq];
      b1val += phi[flocal_1 * ndof * nq + qp1[iq] * ndof + i] * flux[iq];
    }
    atomicAdd(&b0[i], b0val);
    atomicAdd(&b1[i], b1val);
  }
}

// Host wrapper: advance u_n by one explicit Euler convection step.
// Launches compute_w_at_qp, dg0_convection, and dg0_mass in sequence,
// with cudaDeviceSynchronize() between each launch.
//
// w_q is allocated internally as a temporary of size w.size() (one 3-vector
// per cell per local facet). If w is time-independent this could be hoisted.
template <typename ContainerT, typename ContainerI>
void run_dg1_convection(ContainerT& b, ContainerT& u_n, const ContainerT& w,
                        const ContainerT& phi, const ContainerT& normals,
                        const ContainerT& detJ, const ContainerI& facet_to_cell,
                        const ContainerI& facets, const ContainerI& cells,
                        double dt)
{
  using T = typename ContainerT::value_type;

  // Choose a good block size
  dim3 block_size(512);
  dim3 grid_size(facets.size() / block_size.x + 1);

  std::cout << "call dg1_convection:" << facets.size() << "\n";

  dg1_convection<T><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w.data().get(), phi.data().get(),
      normals.data().get(), facet_to_cell.data().get(), facets.data().get(),
      facets.size());

  cudaDeviceSynchronize();
}
