
#include "geometry.h"

// Compute facet normal/jacobians on each facet

template <typename T>
thrust::device_vector<T>
compute_facet_normals(dolfinx::mesh::Mesh<double>& mesh,
                      std::span<const T> dphi)
{
  int tdim = mesh.topology()->dim();
  mesh.topology()->create_connectivity(tdim, tdim - 1);
  auto c_to_f = mesh.topology()->connectivity(tdim, tdim - 1);

  int num_cells = mesh.topology()->index_map(tdim)->size_local();
  int num_facets = mesh.topology()->index_map(tdim - 1)->size_local();
  constexpr int num_facets_per_cell = 4;
  std::vector<std::int32_t> facet_list(num_cells * num_facets_per_cell, -1);
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
        facet_list[idx] = fidx;
      }
    }
  }

  num_cells = 1;
  std::vector<T> facet_jacobians(num_facets * 3, 0.0);
  auto xgeom = mesh.geometry().x();
  auto dofmap = mesh.geometry().dofmap();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get cell geometry
    T coord_dofs[4][3];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 3; ++j)
      {
        coord_dofs[i][j] = xgeom[dofmap(c, i) * 3 + j];
        std::cout << "coord[" << i << "," << j << "]=" << coord_dofs[i][j]
                  << "\n";
      }

    // One quadrature point per facet for now...
    for (int f = 0; f < num_facets_per_cell; ++f)
    {
      std::int32_t idx = facet_list[c * num_facets_per_cell + f];
      if (idx >= 0)
      {
        T J[3][3];
        for (int i = 0; i < 3; i++)
        {
          for (int j = 0; j < 3; j++)
          {
            J[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
              J[i][j] += coord_dofs[k][i] * dphi[j * 16 + f * 4 + k];
          }
        }

        for (int i = 0; i < 3; ++i)
        {
          std::cout << "J=[";
          for (int j = 0; j < 3; ++j)
            std::cout << J[i][j] << " ";
          std::cout << "]\n";
        }

        typedef struct Vec3
        {
          T x, y, z;
        } Vec3;

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
          std::cout << "n[" << i << "] = " << n[i].x << "," << n[i].y << ","
                    << n[i].z << "\n";

        facet_jacobians[idx * 3] = n[f].x;
        facet_jacobians[idx * 3 + 1] = n[f].y;
        facet_jacobians[idx * 3 + 2] = n[f].z;
      }
    }
  }

  for (int f = 0; f < num_facets; ++f)
  {
    std::cout << "[";
    for (int j = 0; j < 3; ++j)
      std::cout << facet_jacobians[f * 3 + j] << " ";
    std::cout << "]\n";
  }

  return thrust::device_vector<T>(facet_jacobians.begin(),
                                  facet_jacobians.end());
}

template thrust::device_vector<double>
compute_facet_normals(dolfinx::mesh::Mesh<double>& mesh,
                      std::span<const double> dphi);
