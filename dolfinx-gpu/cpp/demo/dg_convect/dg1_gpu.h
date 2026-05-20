
// GPU kernels for explicit DG1 upwind convection on tetrahedral meshes.
//
//   dg1_convection   – accumulate upwind flux into b using 6-point
//                      Strang-Fix quadrature on each interior facet
//

#pragma once
#include <cstdint>
#include <thrust/device_vector.h>

// HIP requires the runtime header to be included explicitly for device
// built-ins (blockIdx, blockDim, threadIdx, atomicAdd, etc.).
// CUDA/nvcc injects these implicitly, so no include is needed there.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

// Accumulate the upwind convective flux across each interior facet into b
// using 6-point Strang-Fix quadrature.
//
// For the two cells c0, c1 sharing a facet (c0 < c1, c0 owns the outward
// normal), the upwind flux at each quadrature point iq is:
//   flux[iq] = max(w0.n, 0)*u_n[c0] + min(w1.n, 0)*u_n[c1]
// and the contribution to each DoF i is integrated as:
//   b[c0*ndof+i] -= sum_iq phi[...] * flux[iq]
//   b[c1*ndof+i] += sum_iq phi[...] * flux[iq]
// where w0, w1 and u_n[c0/c1] are evaluated from the DG1 basis functions at
// each quadrature point.  Facet orientation is handled by permuting the
// quadrature-point indices via the qperm table before indexing into phi.
//
// @param b             RHS accumulator, length num_cells*ndof. Zero before
//                      launch.
// @param u_n           DG1 solution coefficients at the previous time step,
//                      layout [cell*ndof + dof].
// @param w             DG1 velocity coefficients, layout
//                      [(cell*ndof + dof)*3 + component].
// @param phi           Basis function values at the 6 Strang-Fix quadrature
//                      points on each local facet, layout
//                      [local_facet * ndof * nq + qp * ndof + dof].
//                      Size: 4 facets * 4 dofs * 6 points = 96 values.
// @param normals       Scaled outward facet normals, layout [facet*3+d].
//                      Magnitude encodes the facet area; points away from c0.
// @param facet_to_cell Layout [facet*2]: packed {(c0<<8)|(perm0<<2)|lf0,
//                      (c1<<8)|(perm1<<2)|lf1}, c0 < c1.
//                      Cell index = value >> 8; permutation = (value>>2)&0x3F;
//                      local facet = value & 0x03.
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

template <typename T>
__global__ void dg1_uwgradv(T* b, const T* u_n, const T* w, const T* detJ,
                            const T* Kadj, const int* cells, int n_cells)
{
  // Load a set of facets
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;
  int cglobal = cells[idx];

  // Four point quadrature in cell (qwts=1/24)
  constexpr int nq = 4;
  constexpr int ndof = 4;
  constexpr T phi[nq][ndof] = {{0.1381966011250091, 0.5854101966249688,
                                0.1381966011250109, 0.1381966011250109},
                               {0.1381966011250091, 0.138196601125011,
                                0.585410196624969, 0.1381966011250109},
                               {0.1381966011250091, 0.138196601125011,
                                0.138196601125011, 0.585410196624969},
                               {0.585410196624967, 0.1381966011250109,
                                0.1381966011250109, 0.1381966011250109}};
  constexpr T dphi[3][nq][ndof] = {};

  for (int iq = 0; iq < nq; ++iq)
  {
    // u and w at quadrature points
    T u0 = 0;
    T w0[3] = {0};
    for (int i = 0; i < ndof; ++i)
    {
      const T* wcell = w + (cglobal * ndof + i) * 3;
      w0[0] += wcell[0] * phi[iq][i];
      w0[1] += wcell[1] * phi[iq][i];
      w0[2] += wcell[2] * phi[iq][i];
      u0 += u_n[cglobal * ndof + i] * phi[iq][i];
    }

    // Geometric transform with J^-T (assumed cellwise constant)
    const T* K = Kadj + cglobal * 9;
    T wr[3];
    wr[0] = u0 * (K[0] * w0[0] + K[1] * w0[1] + K[2] * w0[2]);
    wr[1] = u0 * (K[3] * w0[0] + K[4] * w0[1] + K[5] * w0[2]);
    wr[2] = u0 * (K[6] * w0[0] + K[7] * w0[1] + K[8] * w0[2]);
  }

  // TODO: Multiply by grad(v) - need grad(phi) etc.
  // insert into b
}

// Launches dg1_convection with cudaDeviceSynchronize() after the kernel.
//
// Uses 6-point Strang-Fix quadrature on each interior facet; phi must be
// precomputed with the corresponding quadrature rule (layout described in
// dg1_convection above).
template <typename ContainerT, typename ContainerI>
void run_dg1_convection(ContainerT& b, ContainerT& u_n, const ContainerT& w,
                        const ContainerT& phi, const ContainerT& normals,
                        const ContainerT& detJ, const ContainerT& G,
                        const ContainerI& facet_to_cell,
                        const ContainerI& facets, const ContainerI& cells,
                        double dt)
{
  using T = typename ContainerT::value_type;

  // Choose a good block size
  dim3 block_size(512);
  dim3 grid_size(facets.size() / block_size.x + 1);

  // upwind flux - inner(2 * avg(lmbda * w * u_n), jump(v, n)) * dS
  dg1_convection<T><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w.data().get(), phi.data().get(),
      normals.data().get(), facet_to_cell.data().get(), facets.data().get(),
      facets.size());

  cudaDeviceSynchronize();

  // inner(w*u, grad(v))*dx
  grid_size.x = (cells.size() / block_size.x + 1);
  dg1_uwgradv<T><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w.data().get(), detJ.data().get(),
      G.data().get(), cells.data().get(), cells.size());
}
