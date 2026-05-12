
/// @brief DG0 convection kernel
/// @param b Output field
/// @param u_n Input previous field
/// @param w Velocity vector 3D
/// @param normals Facet normals (incl jacobian scaling)
/// @param facet_to_cell (map from facet to cells)
/// @param facets List of facets to use
/// @param n_facets Length of facet list

template <typename T>
__global__ void dg0_convection(T* b, const T* u_n, const T* w, const T* normals,
                               const std::int32_t* facet_to_cell,
                               const int* facets, int n_facets)
{
  // Load a set of facets
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > n_facets)
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

  atomicAdd(&b[c0], flux);
  atomicAdd(&b[c1], -flux);
}
