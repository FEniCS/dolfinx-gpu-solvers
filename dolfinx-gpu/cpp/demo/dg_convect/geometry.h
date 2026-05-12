
/// Compute detJ at quadrature points of cells
/// For each cell, list of facets to compute detJ/normal on (or -1 if not
/// required).

/// @param N_entity [out] Normals to facets (scaled) at quadrature points
/// @param xgeom Physical geometry points of mesh
/// @param geometry_dofmap Cell geometry dofmap
/// @param dphi_facet Basis function derivatives at quadrature points on facets
/// @param weights Quadrature weights (to match dphi_facet)
/// @param entities List of cells to compute for
/// @param entity_facets List of facets on cell to compute for (or -1 if not,
/// e.g. 4 per cell for tet)
/// @param n_entities Number of cells
/// @param nq Number of quadrature points in dphi_facet per facet (e.g. 1 for
/// linear tet).
/// @param ncdofs Number of coordinate dofs in cell (4 for linear tet).
/// Launch with block size of num_facets_per_cell x nq (e.g. 4 for linear tet).
template <typename T>
__global__ void facet_geometry_computation(
    T* N_entity, const T* xgeom, const std::int32_t* geometry_dofmap,
    const T* dphi_facet, const T* weights, const int* entities,
    const int* entity_facets, int n_entities, int nq, int ncdofs)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Look for facets
  constexpr int num_facets_per_cell = 4;
  const int* facets = entity_facets[c * num_facets_per_cell];

  // Geometric dimension
  constexpr int gdim = 3;

  // Size ncdofs*gdim
  extern __shared__ T _coord_dofs[];

  // First collect cell geometry into shared memory
  int iq = threadIdx.x;
  if (iq >= nq * num_facets_per_cell)
    return;

  if (nq * num_facets_per_cell < ncdofs)
  {
    // Load everything using thread 0
    if (iq == 0)
      for (int i = 0; i < ncdofs; ++i)
        for (int j = 0; j < gdim; ++j)
          _coord_dofs[i * gdim + j]
              = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }
  else if (iq < ncdofs)
  {
    // Use all threads to load
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
    auto _dphi = [&dphi, nq, ncdofs, iq](int i, int j) -> const T
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

    typedef struct Vec3
    {
      T x, y, z;
    };

    auto cross = [](Vec3 a, Vec3 b) -> Vec3
    {
      return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
    };

    Vec3 g1 = {J[0][0], J[1][0], J[2][0]};
    Vec3 g2 = {J[0][1], J[1][1], J[2][1]};
    Vec3 g3 = {J[0][2], J[1][2], J[2][2]};

    Vec3 n[4];
    n[0] = cross(g3, g1);
    n[1] = cross(g2, g1);
    n[2] = cross(g3, g2);
    n[3] = {n[0].x + n[1].x + n[2].x, n[0].y + n[1].y + n[2].y,
            n[0].z + n[1].z + n[2].z};

    for (int i = 0; i < 4; ++i)
    {
      int f_offset = 3 * entity_facets[c * 4];
      if (f >= 0)
      {
        N_entity[f_offset] = n[i].x;
        N_entity[f_offset + 1] = n[i].y;
        N_entity[f_offset + 2] = n[i].z;
      }
    }
  }
}
