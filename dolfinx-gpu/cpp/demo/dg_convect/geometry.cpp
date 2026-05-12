
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

        using Vec3 = std::array<T, 3>;

        auto cross = [](Vec3 a, Vec3 b) -> Vec3
        {
          return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]};
        };

        // Columns of J: images of the reference basis vectors xi_0, xi_1, xi_2.
        // For a P1 tet these are the edge vectors g1=v1-v0, g2=v2-v0, g3=v3-v0.
        Vec3 g1 = {J[0][0], J[1][0], J[2][0]};
        Vec3 g2 = {J[0][1], J[1][1], J[2][1]};
        Vec3 g3 = {J[0][2], J[1][2], J[2][2]};

        // Outward scaled normals for each facet (facet i is opposite vertex i):
        //   Facet 0 (v1,v2,v3): tangents (g2-g1),(g3-g1)  -> (g2-g1) x (g3-g1)
        //   Facet 1 (v0,v2,v3): tangents g2, g3            -> g3 x g2
        //   Facet 2 (v0,v1,v3): tangents g1, g3            -> g1 x g3
        //   Facet 3 (v0,v1,v2): tangents g1, g2            -> g2 x g1
        Vec3 dg21 = {g2[0] - g1[0], g2[1] - g1[1], g2[2] - g1[2]};
        Vec3 dg31 = {g3[0] - g1[0], g3[1] - g1[1], g3[2] - g1[2]};

        std::array<Vec3, 4> n;
        n[0] = cross(dg21, dg31);
        n[1] = cross(g3, g2);
        n[2] = cross(g1, g3);
        n[3] = cross(g2, g1);

        for (int i = 0; i < 4; ++i)
          std::cout << "n[" << i << "] = " << n[i][0] << "," << n[i][1] << ","
                    << n[i][2] << "\n";

        facet_jacobians[idx * 3]     = n[f][0];
        facet_jacobians[idx * 3 + 1] = n[f][1];
        facet_jacobians[idx * 3 + 2] = n[f][2];
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
