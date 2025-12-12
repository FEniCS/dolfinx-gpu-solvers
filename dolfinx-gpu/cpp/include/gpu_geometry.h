
#pragma once

#include <assert.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <cstdint>
#include <cstdio>
#include <dolfinx/mesh/Geometry.h>

namespace
{
/// @brief Computes weighted 3x3 symmetric geometry tensor G from the
/// coordinates and quadrature weights.
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] xgeom Geometry points [*, 3]
/// @param [in] geometry_dofmap Location of coordinates for each cell in
/// xgeom [*, ncdofs]
/// @param [in] dphi Basis derivative tabulation for cell at quadrature
/// points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @param [in] entities list of cells to compute for [n_entities]
/// @param [in] n_entities total number of cells to compute for
/// @param [in] nq number of quadrature points per cell
/// @tparam T scalar type
template <typename T>
__global__ void geometry_computation_G6(T* G_entity, const T* xgeom,
                                        const std::int32_t* geometry_dofmap,
                                        const T* dphi, const T* weights,
                                        const int* entities, int n_entities,
                                        int nq)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of coordinate dofs
  constexpr int ncdofs = 4;

  // Geometric dimension
  constexpr int gdim = 3;

  __shared__ T _coord_dofs[ncdofs * gdim];

  // First collect geometry into shared memory
  int iq = threadIdx.x;
  if (iq >= nq)
    return;

  if (nq < ncdofs)
  {
    if (iq == 0)
      for (int i = 0; i < ncdofs; ++i)
        for (int j = 0; j < gdim; ++j)
          _coord_dofs[i * gdim + j]
              = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }
  else if (iq < ncdofs)
  {
    for (int j = 0; j < 3; ++j)
      _coord_dofs[iq * 3 + j]
          = xgeom[3 * geometry_dofmap[cell * ncdofs + iq] + j];
  }
  __syncthreads();
  // One quadrature point per thread

  // Jacobian
  T J[3][3];
  auto coord_dofs
      = [](int i, int j) -> T& { return _coord_dofs[i * gdim + j]; };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, ncdofs]
    auto _dphi = [&dphi, nq, iq](int i, int j) -> const T
    { return dphi[((i + 1) * nq + iq) * ncdofs + j]; };
    for (std::size_t i = 0; i < gdim; i++)
    {
      for (std::size_t j = 0; j < gdim; j++)
      {
        J[i][j] = 0.0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs(k, i) * _dphi(j, k);
      }
    }
    // Components of K = J^-1 (detJ)
    T K[3][3] = {{J[1][1] * J[2][2] - J[1][2] * J[2][1],
                  -J[0][1] * J[2][2] + J[0][2] * J[2][1],
                  J[0][1] * J[1][2] - J[0][2] * J[1][1]},
                 {-J[1][0] * J[2][2] + J[1][2] * J[2][0],
                  J[0][0] * J[2][2] - J[0][2] * J[2][0],
                  -J[0][0] * J[1][2] + J[0][2] * J[1][0]},
                 {J[1][0] * J[2][1] - J[1][1] * J[2][0],
                  -J[0][0] * J[2][1] + J[0][1] * J[2][0],
                  J[0][0] * J[1][1] - J[0][1] * J[1][0]}};

    T detJ
        = std::abs(J[0][0] * K[0][0] + J[0][1] * K[1][0] + J[0][2] * K[2][0]);

    int offset = (c * nq * 6 + iq);
    G_entity[offset]
        = (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + nq]
        = (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 2 * nq]
        = (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 3 * nq]
        = (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 4 * nq]
        = (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 5 * nq]
        = (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2])
          * weights[iq] / detJ;
  }
}

//-----------------------------------------------------------------------------
template <typename T>
__global__ void geometry_computation_detJ(T* G_entity, const T* xgeom,
                                          const std::int32_t* geometry_dofmap,
                                          const T* dphi, const T* weights,
                                          const int* entities, int n_entities,
                                          int nq)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of coordinate dofs
  constexpr int ncdofs = 4;

  // Geometric dimension
  constexpr int gdim = 3;

  __shared__ T _coord_dofs[ncdofs * gdim];

  // First collect geometry into shared memory
  int iq = threadIdx.x;
  if (iq >= nq)
    return;

  if (nq < ncdofs)
  {
    if (iq == 0)
      for (int i = 0; i < ncdofs; ++i)
        for (int j = 0; j < gdim; ++j)
          _coord_dofs[i * gdim + j]
              = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }
  else if (iq < ncdofs)
  {
    for (int j = 0; j < 3; ++j)
      _coord_dofs[iq * 3 + j]
          = xgeom[3 * geometry_dofmap[cell * ncdofs + iq] + j];
  }
  __syncthreads();
  // One quadrature point per thread

  // Jacobian
  T J[3][3];
  auto coord_dofs
      = [](int i, int j) -> T& { return _coord_dofs[i * gdim + j]; };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, ncdofs]
    auto _dphi = [&dphi, nq, iq](int i, int j) -> const T
    { return dphi[((i + 1) * nq + iq) * ncdofs + j]; };
    for (std::size_t i = 0; i < gdim; i++)
    {
      for (std::size_t j = 0; j < gdim; j++)
      {
        J[i][j] = 0.0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs(k, i) * _dphi(j, k);
      }
    }
    // Cofactors of J
    T K0[3] = {J[1][1] * J[2][2] - J[1][2] * J[2][1],
               -J[1][0] * J[2][2] + J[1][2] * J[2][0],
               J[1][0] * J[2][1] - J[1][1] * J[2][0]};

    T detJ = std::abs(J[0][0] * K0[0] + J[0][1] * K0[1] + J[0][2] * K0[2]);

    int offset = (c * nq + iq);
    G_entity[offset] = detJ * weights[iq];
  }
}

} // namespace

template <typename T>
concept FPholder = requires {
  typename T::value_type;
  requires std::is_floating_point_v<
      typename T::value_type>; // Constrains the value_type to be floating point
};

/// On-device storage of geometry data, used to compute geometric factor at
/// quadrature points
template <FPholder ContainerT, typename ContainerI>
class GPUGeometry
{
  using T = ContainerT::value_type;

public:
  /// @brief Create on-device geometry data at quadrature points
  /// @param geom Input Geometry object from dolfinx
  /// @param qdegree Quadrature degree (must match finite element to be used in
  /// integration)
  GPUGeometry(const dolfinx::mesh::Geometry<T>& geom, int qdegree);

  void set_weights();

  /// Number of quadrature points
  std::size_t num_qp() { return nq; }

  /// Compute the 6 entries of the symmetric geometry tensor
  /// @param[out] G6_q Values at quadrature points
  /// @param[in] cells List of cells to compute on
  void compute_G6(ContainerT& G6_q, const ContainerI& cells);

  /// Compute the determinant of the geometry jacobian at quadrature points
  /// @param[out] detJ_q Determinant of the Jacobian at quadrature points
  /// @param[in] cells List of cells to compute on
  void compute_detJ(ContainerT& detJ_q, const ContainerI& cells);

private:
  // Geometry coordinates
  ContainerT geom_x;

  // Flattened geometry dofmap
  ContainerI geom_dofmap;

  // Quadrature weights
  ContainerT weights;

  // Basis function and derivatives for geometry
  ContainerT phi_data;

  // Number of quadrature points per cell
  std::size_t nq;
};

/// Initialize
template <FPholder ContainerT, typename ContainerI>
GPUGeometry<ContainerT, ContainerI>::GPUGeometry(
    const dolfinx::mesh::Geometry<typename ContainerT::value_type>& geometry,
    int qdegree)
{
  // Coordinate Element
  auto element = geometry.cmap();

  // Copy geometry data to device
  geom_x.assign(geometry.x().data(), geometry.x().data() + geometry.x().size());
  geom_dofmap.assign(geometry.dofmap().data_handle(),
                     geometry.dofmap().data_handle()
                         + geometry.dofmap().size());

  // Set up quadrature weights and basis functions
  basix::cell::type cell_type = cell_type_to_basix_type(element.cell_shape());
  auto [qpoints, qweights] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::Default, cell_type,
      basix::polyset::type::standard, qdegree);
  nq = qweights.size();

  auto shape = element.tabulate_shape(1, nq);
  std::vector<T> table(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  element.tabulate(1, qpoints, {nq, 3}, std::span(table));
  assert(shape.size() == 4);
  assert(shape[0] == 4);
  assert(shape[1] == nq);
  int ndofs = shape[2];
  assert(shape[3] == 1);
  std::cout << "ndofs = " << ndofs << ", nq = " << nq << "\n";

  // Copy basis function and derivatives at qpts to phi_data
  phi_data.assign(table.begin(), table.end());
  weights.assign(qweights.begin(), qweights.end());
}
//--------------------------------------------------------------------------
template <FPholder ContainerT, typename ContainerI>
void GPUGeometry<ContainerT, ContainerI>::compute_G6(ContainerT& G_q,
                                                     const ContainerI& cells)
{
  using T = ContainerT::value_type;

  dim3 block_size_g(nq);
  dim3 grid_size_g(cells.size());
  geometry_computation_G6<T><<<grid_size_g, block_size_g>>>(
      G_q.data().get(), geom_x.data().get(), geom_dofmap.data().get(),
      phi_data.data().get(), weights.data().get(), cells.data().get(),
      cells.size(), nq);
}
//--------------------------------------------------------------------------
template <FPholder ContainerT, typename ContainerI>
void GPUGeometry<ContainerT, ContainerI>::compute_detJ(ContainerT& detJ_q,
                                                       const ContainerI& cells)
{
  using T = ContainerT::value_type;
  dim3 block_size_g(nq);
  dim3 grid_size_g(cells.size());
  geometry_computation_detJ<T><<<grid_size_g, block_size_g>>>(
      detJ_q.data().get(), geom_x.data().get(), geom_dofmap.data().get(),
      phi_data.data().get(), weights.data().get(), cells.data().get(),
      cells.size(), nq);
}
