
#include <thrust/device_vector.hpp>

// Compute facet normal/jacobians on each facet

template <typename T>
void compute_facet_normals(const dolfinx::mesh::Mesh<T>& mesh)
{
  int tdim = mesh.topology().dim();
  mesh.topology().create_connectivity(tdim, tdim - 1);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);

  int num_cells = mesh.topology().index_map(tdim)->size_local();
  constexpr int num_facets_per_cell = 4;
  std::vector<std::int32_t> facet_list(num_cells * num_facets_per_cell, -1);
  std::vector<bool> facet_tick(num_facets, false);

  // Tick off facets in increasing cell index order, ensuring lowest numbered
  // cells always do the facet computation.
  for (int i = 0; i < num_cells; ++i)
  {
    auto facets = c_to_f.links(i);
    for (std::int32_t f = 0; f < num_facets_per_cell; ++f)
    {
      std::int32_t fidx = facets[f];
      if (facet_tick[fidx] == false)
      {
        facet_tick[fidx] = true;
        std::int32_t idx = i * num_facets_per_cell + f;
        facet_list[idx] = fidx;
      }
    }
  }
}
