
#include <dolfinx/mesh/Mesh.h>
#include <thrust/device_vector.h>

template <typename T>
thrust::device_vector<T>
compute_facet_normals(dolfinx::mesh::Mesh<double>& mesh,
                      std::span<const T> dphi);
