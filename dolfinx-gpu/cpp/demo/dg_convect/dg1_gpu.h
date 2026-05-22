
// GPU kernels for explicit DG1 upwind convection on tetrahedral meshes.
//
//   dg1_convection   – accumulate upwind flux into b using 6-point
//                      Strang-Fix quadrature on each interior facet
//

#pragma once
#include <cstdint>
#include <dolfinx/common/Timer.h>
#include <thrust/device_vector.h>

// HIP requires the runtime header to be included explicitly for device
// built-ins (blockIdx, blockDim, threadIdx, atomicAdd, etc.).
// CUDA/nvcc injects these implicitly, so no include is needed there.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#define gpuError_t hipError_t
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#else
#define gpuError_t cudaError_t
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
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
//                      layout [(cell*ndof + dof)*nc + component].
// @param nc            Number of components in u_n
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
template <typename T, int nc>
__global__ void dg1_convection(T* b, const T* u_n, const T* w, const T* phi,
                               const T* normals,
                               const std::int32_t* facet_to_cell,
                               const int* facets, int n_facets)
{
  // Load a set of facets
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Number of DoFs per cell (DG1)
  constexpr int ndof = 4;
  // Number of quadrature points per facet
  constexpr int nq = 6;

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

  const T* n = normals + fglobal * 3;

  T b0val[ndof * nc] = {0};
  T b1val[ndof * nc] = {0};
  for (int iq = 0; iq < nq; ++iq)
  {
    // Compute w.n at quadrature points on facet of cell0
    T w0n = 0;
    // Get u value at quadrature points on facet, cell0
    const T* u0 = u_n + (c0 * ndof * nc);
    T uf0[nc] = {0};
    for (int i = 0; i < ndof; ++i)
    {
      const T* w0 = w + (c0 * ndof + i) * 3;
      T phi_i = phi[flocal_0 * ndof * nq + qp0[iq] * ndof + i];
      w0n += w0[0] * phi_i * n[0];
      w0n += w0[1] * phi_i * n[1];
      w0n += w0[2] * phi_i * n[2];
      for (int c = 0; c < nc; ++c)
        uf0[c] += u0[i * nc + c] * phi_i;
    }

    // Compute w.n at quadrature points on facet of cell1
    T w1n = 0;
    // Get u value at quadrature points on facet, cell1
    const T* u1 = u_n + (c1 * ndof * nc);
    T uf1[nc] = {0};
    for (int i = 0; i < ndof; ++i)
    {
      const T* w1 = w + (c1 * ndof + i) * 3;
      T phi_i = phi[flocal_1 * ndof * nq + qp1[iq] * ndof + i];
      w1n += w1[0] * phi_i * n[0];
      w1n += w1[1] * phi_i * n[1];
      w1n += w1[2] * phi_i * n[2];
      for (int c = 0; c < nc; ++c)
        uf1[c] += u1[i * nc + c] * phi_i;
    }

    // Apply upwinding at quadrature points
    for (int c = 0; c < nc; ++c)
    {
      T flux = fmax(w0n, 0) * uf0[c] + fmin(w1n, 0) * uf1[c];
      for (int i = 0; i < ndof; ++i)
      {
        b0val[i * nc + c]
            -= phi[flocal_0 * ndof * nq + qp0[iq] * ndof + i] * flux;
        b1val[i * nc + c]
            += phi[flocal_1 * ndof * nq + qp1[iq] * ndof + i] * flux;
      }
    }
  }

  // Locate b cell dofs in output data
  // Integrate at quadrature points (weight = 1/12)
  T* b0 = b + c0 * ndof * nc;
  T* b1 = b + c1 * ndof * nc;
  for (int ic = 0; ic < ndof * nc; ++ic)
  {
    b0val[ic] /= T(12);
    b1val[ic] /= T(12);
    atomicAdd(&b0[ic], b0val[ic]);
    atomicAdd(&b1[ic], b1val[ic]);
  }
}

// Accumulate the cell-volume term  inner(u*w, grad(v)) * dx  into b.
//
// This is the integration-by-parts counterpart of the facet flux in
// dg1_convection.  Together they discretise the advection operator
//   -inner(u*w, grad(v))*dx + upwind_flux(u, v)*dS
// in the DG1 strong form.
//
// Uses a 4-point symmetric quadrature rule (weight = 1/24 each) exact
// for polynomials of degree ≤ 2 on a tetrahedron.
//
// @param b      RHS accumulator, length num_cells*ndof.
// @param u_n    DG1 solution coefficients, layout [cell*ndof + dof].
// @param w      DG1 velocity coefficients, layout [(cell*ndof + dof)*3 + d].
// @param Kadj   Adjugate Jacobian adj(J) per cell, layout [cell*9 + row*3+col]
//               (row-major 3×3).  Used to map reference gradients to physical
//               space: grad_x phi_i = Kadj * grad_xi phi_i / det(J), but since
//               the 1/det(J) cancels with the physical volume element the
//               adjugate is used directly.
// @param cells  Cell indices to process.
// @param n_cells Length of cells.
template <typename T, int nc>
__global__ void dg1_uwgradv(T* b, const T* u_n, const T* w, const T* Kadj,
                            const int* cells, int n_cells)
{
  // Load a set of cells
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;
  int cglobal = cells[idx];

  // Four point quadrature in cell (qwts=1/24)
  // TODO: pass phi/dphi as a parameter
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
  constexpr T dphi[3][ndof] = {{-1, 1, 0, 0}, {-1, 0, 1, 0}, {-1, 0, 0, 1}};

  T bval[ndof * nc] = {0};
  for (int iq = 0; iq < nq; ++iq)
  {
    // u and w at quadrature points
    T u0[nc] = {0};
    T w0[3] = {0};
    for (int i = 0; i < ndof; ++i)
    {
      const T* wcell = w + (cglobal * ndof + i) * 3;
      w0[0] += wcell[0] * phi[iq][i];
      w0[1] += wcell[1] * phi[iq][i];
      w0[2] += wcell[2] * phi[iq][i];
      for (int c = 0; c < nc; ++c)
        u0[c] += u_n[(cglobal * ndof + i) * nc + c] * phi[iq][i];
    }

    // Geometric transform with K=adj(J) (assumed cellwise constant)
    const T* K = Kadj + cglobal * 9;
    T wr[3] = {0};
    wr[0] = (K[0] * w0[0] + K[1] * w0[1] + K[2] * w0[2]);
    wr[1] = (K[3] * w0[0] + K[4] * w0[1] + K[5] * w0[2]);
    wr[2] = (K[6] * w0[0] + K[7] * w0[1] + K[8] * w0[2]);

    for (int i = 0; i < ndof; ++i)
    {
      for (int c = 0; c < nc; ++c)
      {
        bval[i * nc + c]
            += (wr[0] * dphi[0][i] + wr[1] * dphi[1][i] + wr[2] * dphi[2][i])
               * u0[c];
      }
    }
  }

  for (int ic = 0; ic < ndof * nc; ++ic)
  {
    bval[ic] /= T(24);
    atomicAdd(&b[cglobal * ndof * nc + ic], bval[ic]);
  }
}

// Launches dg1_convection with cudaDeviceSynchronize() after the kernel.
//
// Uses 6-point Strang-Fix quadrature on each interior facet; phi must be
// precomputed with the corresponding quadrature rule (layout described in
// dg1_convection above).
template <typename ContainerT, typename ContainerI>
void run_dg1_convection(ContainerT& b, ContainerT& u_n, const ContainerT& w,
                        const ContainerT& phi, const ContainerT& normals,
                        const ContainerT& Kadj, const ContainerI& facet_to_cell,
                        const ContainerI& facets, const ContainerI& cells)
{
  using T = typename ContainerT::value_type;

  dolfinx::common::Timer tsolve("*DG: Compute advection");

  // Choose a good block size
  dim3 block_size(512);
  dim3 grid_size(facets.size() / block_size.x + 1);

  // number of components in u_n = 1

  // upwind flux - inner(2 * avg(lmbda * w * u_n), jump(v, n)) * dS
  dg1_convection<T, 3><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w.data().get(), phi.data().get(),
      normals.data().get(), facet_to_cell.data().get(), facets.data().get(),
      facets.size());

  gpuError_t err = gpuDeviceSynchronize();
  if (err != gpuSuccess)
    printf("kernel launch failed with error \"%s\".\n", gpuGetErrorString(err));

  // inner(w*u, grad(v))*dx  (volume term, balances the facet flux above)
  grid_size.x = (cells.size() / block_size.x + 1);
  dg1_uwgradv<T, 3><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w.data().get(), Kadj.data().get(),
      cells.data().get(), cells.size());

  err = gpuDeviceSynchronize();
  if (err != gpuSuccess)
    printf("kernel launch failed with error \"%s\".\n", gpuGetErrorString(err));
}
