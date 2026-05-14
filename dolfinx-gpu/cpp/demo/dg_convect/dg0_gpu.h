
#pragma once
#include <cstdint>
#include <thrust/device_vector.h>

// HIP requires the runtime header to be included explicitly for device
// built-ins (blockIdx, blockDim, threadIdx, atomicAdd, etc.).
// CUDA/nvcc injects these implicitly, so no include is needed there.
#if defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

template <typename T>
__global__ void compute_w_at_qp(const T* w_dof, const T* phi, T* w_q,
                                const std::int32_t* cell_to_facet,
                                const std::int32_t* cells, int n_cells)
{
  // Load a set of cells
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells)
    return;
  std::int32_t cglobal = cells[idx];

  // For P1, one quadrature point
  constexpr int nq = 1;
  // 4 basis functions on tet
  constexpr int nphi = 4;

  // Get facets of cell
  const std::int32_t* facets = cell_to_facet + cglobal * 4;
  for (int f = 0; f < 4; ++f)
  {
    if (facets[f] != -1)
    {
      for (int iq = 0; iq < nq; ++iq)
      {
        int f0 = facets[f] * nq + iq;
        for (int i = 0; i < nphi; ++i)
        {
          int c0 = cglobal * nphi + i;
          T phi_val = phi[(f * nq + iq) * nphi + i];

          // Compute w at qp
          w_q[f0 * 3] += w_dof[c0 * 3] * phi_val;
          w_q[f0 * 3 + 1] += w_dof[c0 * 3 + 1] * phi_val;
          w_q[f0 * 3 + 2] += w_dof[c0 * 3 + 2] * phi_val;
        }
      }
    }
  }
}

/// @brief DG0 convection kernel
/// @param b Output field
/// @param u_n Input previous field
/// @param w Velocity vector 3D
/// @param phi Basis functions for u_n and w
/// @param normals Facet normals (incl jacobian scaling)
/// @param facet_to_cell (map from facet to cells)
/// @param facets List of facets to use
/// @param n_facets Length of facet list
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
  int c0 = facet_to_cell[fglobal * 2];
  int c1 = facet_to_cell[fglobal * 2 + 1];

  // Compute w.n on both sides of facet
  const T* n = normals + fglobal * 3;
  const T* w0 = w + c0 * 3;
  const T* w1 = w + c1 * 3;
  T w0n = n[0] * w0[0] + n[1] * w0[1] + n[2] * w0[2];
  T w1n = n[0] * w1[0] + n[1] * w1[1] + n[2] * w1[2];

  // Apply upwinding
  T flux = fmax(w0n, 0) * u_n[c0] + fmin(w1n, 0) * u_n[c1];

  atomicAdd(&b[c0], -flux);
  atomicAdd(&b[c1], flux);
}

// Assemble RHS dx term and add to u_n
// L = inner(u_n / dt, v) * dx
// m = inner(1/dt, v) * dx
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

template <typename ContainerT, typename ContainerI>
void run_dg0_convection(ContainerT& b, ContainerT& u_n, const ContainerT& w,
                        const ContainerT& phi, const ContainerT& normals,
                        const ContainerT& detJ, const ContainerI& facet_to_cell,
                        const ContainerI& cell_to_facet,
                        const ContainerI& facets, const ContainerI& cells,
                        double dt)
{
  using T = typename ContainerT::value_type;

  // Choose a good block size
  dim3 block_size(512);
  dim3 grid_size(facets.size() / block_size.x + 1);

  // If w is constant, could move this outside loop
  constexpr int nq = 1;
  thrust::device_vector<T> w_q(facets.size() * 3 * nq, T(0));
  grid_size = dim3(cells.size() / block_size.x + 1);
  compute_w_at_qp<T><<<grid_size, block_size>>>(
      w.data().get(), phi.data().get(), w_q.data().get(),
      cell_to_facet.data().get(), cells.data().get(), cells.size());

  cudaSynchronizeDevice();

  grid_size = dim3(facets.size() / block_size.x + 1);
  dg0_convection<T><<<grid_size, block_size>>>(
      b.data().get(), u_n.data().get(), w_q.data().get(), phi.data().get(),
      normals.data().get(), facet_to_cell.data().get(), facets.data().get(),
      facets.size());

  cudaSynchronizeDevice();

  grid_size = dim3(cells.size() / block_size.x + 1);
  dg0_mass<T><<<grid_size, block_size>>>(u_n.data().get(), dt, b.data().get(),
                                         detJ.data().get(), cells.data().get(),
                                         cells.size());
}
