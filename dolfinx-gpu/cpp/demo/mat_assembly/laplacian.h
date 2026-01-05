
#pragma once

#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <dolfinx/la/MatrixCSR.h>

namespace detail
{
/// @brief Assemble element matrix for Laplacian operator into global tensor
/// @param Avals Values array of CSR matrix to assemble into
/// @param Acols Columns array of CSR matrix to assemble into
/// @param Arowptr Row pointer array of CSR matrix to assemble into
/// @param phi_data Gradient of basis at quadrature points
/// @param G_entity Precomputed geometric tensor at quadrature points
/// @param cell_dofs List of dofs on each cell
/// @param cells List of cells
/// @param ncells Number of cells in list
/// @param nq Number of quadrature points per cell
/// @param ndofs Number of dofs per cell
/// @tparam T Scalar type
template <typename T>
__global__ void laplacian_matinsert(
    T* __restrict__ Avals, const std::int32_t* __restrict__ Acols,
    const std::int32_t* __restrict__ Arowptr, const T* __restrict__ phi_data,
    const T* __restrict__ G_entity, const std::int32_t* __restrict__ cell_dofs,
    const int* __restrict__ cells, int ncells, int nq, int ndofs)
{
  if (blockDim.x != ndofs or blockDim.y != ndofs)
  {
    printf("Incorrect blockDim\n");
    abort();
  }

  if (gridDim.x != ncells)
  {
    printf("Incorrect gridDim.x\n");
    abort();
  }
  if (blockIdx.x >= ncells)
    abort();

  int cell_id = cells[blockIdx.x];

  const int tx = threadIdx.x; // dofs i
  const int ty = threadIdx.y; // dofs j

  // const T* phi = phi_data; - not needed for this kernel
  const T* dphix = phi_data + nq * ndofs;
  const T* dphiy = phi_data + 2 * nq * ndofs;
  const T* dphiz = phi_data + 3 * nq * ndofs;

  T Atmp = 0.0;
  for (int iq = 0; iq < nq; iq++)
  {
    int g_offset = cell_id * nq * 6 + iq;
    T G0 = G_entity[g_offset + 0 * nq];
    T G1 = G_entity[g_offset + 1 * nq];
    T G2 = G_entity[g_offset + 2 * nq];
    T G3 = G_entity[g_offset + 3 * nq];
    T G4 = G_entity[g_offset + 4 * nq];
    T G5 = G_entity[g_offset + 5 * nq];

    int i = iq * ndofs + tx;
    int j = iq * ndofs + ty;

    Atmp += dphix[i] * dphix[j] * G0 + dphix[i] * dphiy[j] * G1
            + dphix[i] * dphiz[j] * G2;
    Atmp += dphiy[i] * dphix[j] * G1 + dphiy[i] * dphiy[j] * G3
            + dphiy[i] * dphiz[j] * G4;
    Atmp += dphiz[i] * dphix[j] * G2 + dphiz[i] * dphiy[j] * G4
            + dphiz[i] * dphiz[j] * G5;
  }

  // Insert into CSR matrix
  // Find location for Atmp in Avals
  int row = cell_dofs[cell_id * ndofs + tx];
  int col = cell_dofs[cell_id * ndofs + ty];

  // Binary search in row
  int minidx = Arowptr[row], maxidx = Arowptr[row + 1];
  while (minidx <= maxidx)
  {
    int idx = minidx + (maxidx - minidx) / 2;
    // int idx = (maxidx + minidx) / 2;
    if (Acols[idx] < col)
      minidx = idx + 1;
    else if (Acols[idx] > col)
      maxidx = idx - 1;
    else // Acols[idx] == col
    {
      atomicAdd(&Avals[idx], Atmp);
      break;
    }
  }
}
} // namespace detail

/// @brief Assemble into CSR matrix
/// @param A Matrix to assemble into
/// @param phi_data Basis evaluation data at reference quadrature points
/// @param G Geometry transform at each quadrature point
/// @param cell_dofs DofMap
/// @param cells List of cells to integrate over
/// @param nq Number of quadrature points per cell
/// @param ndofs Number of dofs on each cell
template <typename T, typename ContainerT, typename ContainerI>
void assemble(dolfinx::la::MatrixCSR<T, ContainerT, ContainerI, ContainerI>& A,
              const ContainerT& phi_data, const ContainerT& G,
              const ContainerI& cell_dofs, const ContainerI& cells, int nq,
              int ndofs)
{
  dim3 block_size(ndofs, ndofs);
  dim3 grid_size(cells.size());

  detail::laplacian_matinsert<T><<<grid_size, block_size>>>(
      A.values().data().get(), A.cols().data().get(), A.row_ptr().data().get(),
      phi_data.data().get(), G.data().get(), cell_dofs.data().get(),
      cells.data().get(), cells.size(), nq, ndofs);
}
