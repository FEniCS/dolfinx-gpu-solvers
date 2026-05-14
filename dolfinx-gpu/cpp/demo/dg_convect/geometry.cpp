
#include "geometry.h"

// Compute facet normal/jacobians on each facet

template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>>
compute_facet_normals(dolfinx::mesh::Mesh<double>& mesh, std::span<const T> phi)
{
  // Extract basis function derivatives from table (phi, phi_x, phi_y, phi_z)
  std::span<const T> dphi(std::next(phi.begin(), phi.size() / 4),
                          phi.size() * 3 / 4);
  int tdim = mesh.topology()->dim();
  mesh.topology()->create_connectivity(tdim, tdim - 1);
  auto c_to_f = mesh.topology()->connectivity(tdim, tdim - 1);

  int num_cells = mesh.topology()->index_map(tdim)->size_local();
  int num_facets = mesh.topology()->index_map(tdim - 1)->size_local();
  constexpr int num_facets_per_cell = 4;
  std::vector<std::int32_t> facet_list0(num_cells * num_facets_per_cell, -1);
  std::vector<bool> facet_tick(num_facets, false);

  // Tick off facets in increasing cell index order, ensuring lowest numbered
  // cells always do the facet computation.
  for (int c = 0; c < num_cells; ++c)
  {
    auto facets = c_to_f->links(c);
    for (std::int32_t f = 0; f < num_facets_per_cell; ++f)
    {
      std::int32_t fidx = facets[f];
      if (facet_tick[fidx] == false)
      {
        facet_tick[fidx] = true;
        std::int32_t idx = c * num_facets_per_cell + f;
        facet_list0[idx] = fidx;
      }
    }
  }

  // Compute facet Jacobians/normals
  std::vector<T> facet_jacobians(num_facets * 3, 0.0);
  std::vector<T> detJ(num_cells);
  auto xgeom = mesh.geometry().x();
  auto dofmap = mesh.geometry().dofmap();

  // Cross product
  using Vec3 = std::array<T, 3>;
  auto cross = [](Vec3 a, Vec3 b) -> Vec3
  {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  };

  // Iterate over cells
  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell geometry
    T coord_dofs[4][3];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 3; ++j)
        coord_dofs[i][j] = xgeom[dofmap(c, i) * 3 + j];

    // For a linear tet, J is constant across the cell. Compute it once using
    // facet f=0's dphi (all facets give the same result).
    T J[3][3];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
        J[i][j] = 0.0;
        for (int k = 0; k < 4; k++)
          J[i][j] += coord_dofs[k][i] * dphi[j * 16 + k]; // f=0
      }

    detJ[c] = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
              - J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2])
              + J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]);

    // Columns of J: images of the reference basis vectors xi_0, xi_1, xi_2.
    // For a P1 tet these are the edge vectors g1=v1-v0, g2=v2-v0, g3=v3-v0.
    Vec3 g1 = {J[0][0], J[1][0], J[2][0]};
    Vec3 g2 = {J[0][1], J[1][1], J[2][1]};
    Vec3 g3 = {J[0][2], J[1][2], J[2][2]};

    // Normal vectors from Jacobian - magnitude is detJ(facet).
    // Computed once per cell; only written to facet_jacobians for owned facets.
    std::array<Vec3, 4> n;
    n[1] = cross(g3, g2);
    n[2] = cross(g1, g3);
    n[3] = cross(g2, g1);
    n[0] = {-n[1][0] - n[2][0] - n[3][0], -n[1][1] - n[2][1] - n[3][1],
            -n[1][2] - n[2][2] - n[3][2]};

    // Write normals only for facets this cell owns
    for (int f = 0; f < num_facets_per_cell; ++f)
    {
      std::int32_t idx = facet_list0[c * num_facets_per_cell + f];
      if (idx >= 0)
      {
        facet_jacobians[idx * 3] = n[f][0];
        facet_jacobians[idx * 3 + 1] = n[f][1];
        facet_jacobians[idx * 3 + 2] = n[f][2];
      }
    }
  }

  return {
      thrust::device_vector<T>(facet_jacobians.begin(), facet_jacobians.end()),
      thrust::device_vector<T>(detJ.begin(), detJ.end())};
}

template std::tuple<thrust::device_vector<double>,
                    thrust::device_vector<double>>
compute_facet_normals(dolfinx::mesh::Mesh<double>& mesh,
                      std::span<const double> dphi);
